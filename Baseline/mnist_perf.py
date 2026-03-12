from __future__ import print_function
import argparse
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

try:
    import wandb
except ImportError:
    wandb = None

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


_AUC_WARNING_EMITTED = False

class Net(nn.Module):
    def __init__(self, width):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, int(32*width), 3, 1)
        self.conv2 = nn.Conv2d(int(32*width), int(64*width), 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(int(9216*width), int(128*width))
        self.fc2 = nn.Linear(int(128*width), 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def init_wandb(args):
    if not args.use_wandb:
        print('W&B disabled. Pass --use-wandb to enable experiment logging.')
        return None

    if wandb is None:
        print('W&B logging requested but wandb is not installed. Install with: pip install wandb')
        return None

    if args.wandb_mode == 'disabled':
        print('W&B disabled via --wandb-mode disabled.')
        return None

    api_key = os.environ.get('WANDB_API_KEY', '') or args.wandb_api_key

    if args.wandb_mode == 'offline':
        print('W&B running in offline mode. Use wandb sync later to upload the run.')

    try:
        if api_key:
            wandb.login(key=api_key)
        elif args.wandb_mode == 'online':
            print('W&B has no API key from --wandb-api-key or WANDB_API_KEY. Online runs may fail unless this machine is already logged in.')
    except Exception as exc:
        print('W&B login failed: {}'.format(exc))
        if args.wandb_mode == 'online':
            return None

    run_config = vars(args).copy()
    run_config.pop('wandb_api_key', None)

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=args.wandb_run_name if args.wandb_run_name else None,
            mode=args.wandb_mode,
            config=run_config,
        )
        print('W&B initialized: project={}, entity={}, mode={}, run_name={}'.format(
            args.wandb_project,
            args.wandb_entity if args.wandb_entity else '<default>',
            args.wandb_mode,
            args.wandb_run_name if args.wandb_run_name else '<auto>',
        ))
        return run
    except Exception as exc:
        print('W&B init failed: {}'.format(exc))
        return None


def log_to_wandb(run, metrics, step=None):
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as exc:
        print('W&B log failed: {}'.format(exc))


def finish_wandb(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print('W&B finish failed: {}'.format(exc))


def compute_multiclass_auc(targets, probabilities):
    global _AUC_WARNING_EMITTED

    if roc_auc_score is None:
        if not _AUC_WARNING_EMITTED:
            print('AUC unavailable: scikit-learn is not installed in this environment.')
            _AUC_WARNING_EMITTED = True
        return float('nan')

    targets_np = targets.cpu().numpy()
    probs_np = probabilities.cpu().numpy()
    num_classes = probabilities.size(1)
    class_aucs = []
    present_classes = sorted(set(int(value) for value in targets_np.tolist()))

    try:
        for class_index in range(num_classes):
            binary_targets = (targets_np == class_index).astype(int)
            positive_count = int(binary_targets.sum())
            negative_count = int(binary_targets.shape[0] - positive_count)

            if positive_count == 0 or negative_count == 0:
                continue

            class_auc = roc_auc_score(binary_targets, probs_np[:, class_index])
            if not math.isnan(float(class_auc)):
                class_aucs.append(float(class_auc))

        if class_aucs:
            return float(sum(class_aucs) / len(class_aucs))

        if not _AUC_WARNING_EMITTED:
            print(
                'AUC unavailable: no valid one-vs-rest class AUCs could be computed. '
                'present_classes={}, probability_shape={}, probability_min={}, probability_max={}'.format(
                    present_classes,
                    probs_np.shape,
                    float(probs_np.min()),
                    float(probs_np.max()),
                )
            )
            _AUC_WARNING_EMITTED = True
        return float('nan')
    except ValueError as exc:
        if not _AUC_WARNING_EMITTED:
            print(
                'AUC unavailable: {}. present_classes={}, probability_shape={}, probability_min={}, probability_max={}'.format(
                    exc,
                    present_classes,
                    probs_np.shape,
                    float(probs_np.min()),
                    float(probs_np.max()),
                )
            )
            _AUC_WARNING_EMITTED = True
        return float('nan')
    except Exception as exc:
        if not _AUC_WARNING_EMITTED:
            print(
                'AUC unavailable due to unexpected error: {}. present_classes={}, probability_shape={}, probability_min={}, probability_max={}'.format(
                    exc,
                    present_classes,
                    probs_np.shape,
                    float(probs_np.min()),
                    float(probs_np.max()),
                )
            )
            _AUC_WARNING_EMITTED = True
        return float('nan')


def benchmark_inference_throughput(model, data_loader, device, max_batches=20):
    model_was_training = model.training
    model.eval()
    total_inputs = 0

    with torch.no_grad():
        warmup_batches = 2
        for idx, (data, _) in enumerate(data_loader):
            if idx >= warmup_batches:
                break
            data = data.to(device)
            _ = model(data)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        for idx, (data, _) in enumerate(data_loader):
            if idx >= max_batches:
                break
            data = data.to(device)
            _ = model(data)
            total_inputs += data.size(0)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if model_was_training:
        model.train()

    if elapsed <= 0:
        return float('nan')
    return float(total_inputs / elapsed)


def benchmark_latency_ms(model, data_loader, device, max_batches=20):
    model_was_training = model.training
    model.eval()
    total_batches = 0

    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()

        for idx, (data, _) in enumerate(data_loader):
            if idx >= max_batches:
                break
            data = data.to(device)
            _ = model(data)
            total_batches += 1

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    if model_was_training:
        model.train()

    if total_batches == 0:
        return float('nan')
    return float((elapsed / total_batches) * 1000.0)


def benchmark_cpu_latency_single_core_ms(model, data_loader, max_batches=20):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        return benchmark_latency_ms(model, data_loader, torch.device('cpu'), max_batches=max_batches)
    finally:
        torch.set_num_threads(previous_threads)


def estimate_flops_with_hooks(model, device, input_shape=(1, 1, 28, 28)):
    total_flops = 0.0
    handles = []
    model_was_training = model.training

    def conv_hook(module, inputs, output):
        nonlocal total_flops
        batch_size = output.shape[0]
        out_channels = output.shape[1]
        out_height = output.shape[2]
        out_width = output.shape[3]
        kernel_height, kernel_width = module.kernel_size
        kernel_mul = (module.in_channels // module.groups) * kernel_height * kernel_width
        bias_ops = 1 if module.bias is not None else 0
        ops_per_output = (2 * kernel_mul) + bias_ops
        total_flops += batch_size * out_channels * out_height * out_width * ops_per_output

    def linear_hook(module, inputs, output):
        nonlocal total_flops
        if output.dim() == 1:
            batch_size = 1
            out_features = output.shape[0]
        else:
            batch_size = output.shape[0]
            out_features = output.shape[-1]
        bias_ops = 1 if module.bias is not None else 0
        ops_per_output = (2 * module.in_features) + bias_ops
        total_flops += batch_size * out_features * ops_per_output

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    sample_input = torch.randn(*input_shape, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(sample_input)

    for handle in handles:
        handle.remove()

    if model_was_training:
        model.train()

    return float(total_flops)


def compute_model_stats(model, device):
    param_count = sum(p.numel() for p in model.parameters())
    flops = float('nan')
    flops_source = 'unavailable'

    if FlopCountAnalysis is not None:
        try:
            sample_input = torch.randn(1, 1, 28, 28, device=device)
            flops = float(FlopCountAnalysis(model, sample_input).total())
            flops_source = 'fvcore'
        except Exception:
            flops = float('nan')

    if math.isnan(flops):
        try:
            flops = estimate_flops_with_hooks(model, device)
            flops_source = 'approximate_hooks'
        except Exception as exc:
            print('FLOPS fallback failed: {}'.format(exc))
            flops = float('nan')

    return param_count, flops, flops_source


def safe_number(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def update_min_max(stats, key, value):
    value = safe_number(value)
    if value is None:
        return
    stats[f'{key}_min'] = min(stats.get(f'{key}_min', value), value)
    stats[f'{key}_max'] = max(stats.get(f'{key}_max', value), value)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0

    #Loop over all the batches in the dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        #Pass the data through your model to get the output
        output = model(data)
        #Calculate the error
        loss = F.nll_loss(output, target)
        #Backpropagate the error through the network
        loss.backward()
        #Modify the weights based on the calculated gradient
        optimizer.step()
        #Display Metrics
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        #Determine the predictions the network was making
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #Increment how many times it was correct
        correct += pred.eq(target.view_as(pred)).sum()
    #Add the new score to the tracker which may restructured the model with PB Nodes
    GPA.pai_tracker.add_extra_score(100. * correct / len(train_loader.dataset), 'train') 
    model.to(device)
    train_accuracy = 100. * correct.item() / len(train_loader.dataset)
    return train_accuracy


def evaluate(model, device, data_loader):
    model.eval()
    avg_loss = 0
    correct = 0
    all_targets = []
    all_probs = []
    #Dont calculate Gradients
    with torch.no_grad():
        #Loop over all the test data
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            #Pass the data through your model to get the output
            output = model(data)
            probs = torch.exp(output)
            #Calculate the error
            avg_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #Determine the predictions the network was making
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #Increment how many times it was correct
            correct += pred.eq(target.view_as(pred)).sum()
            all_targets.append(target.detach().cpu())
            all_probs.append(probs.detach().cpu())

    #Display Metrics
    avg_loss /= len(data_loader.dataset)
    accuracy = 100. * correct.item() / len(data_loader.dataset)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    auc = compute_multiclass_auc(all_targets, all_probs)
    #For single-label classification, Precision@1 equals top-1 accuracy.
    precision_at_1 = accuracy

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'precision_at_1': precision_at_1,
    }
    return metrics


def test(model, device, val_loader, test_loader, optimizer, scheduler, args):
    val_metrics = evaluate(model, device, val_loader)
    test_metrics = evaluate(model, device, test_loader)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {:.2f}%, AUC: {}\n'.format(
        val_metrics['loss'], val_metrics['accuracy'],
        'nan' if safe_number(val_metrics['auc']) is None else '{:.6f}'.format(val_metrics['auc'])))
    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, AUC: {}\n'.format(
        test_metrics['loss'], test_metrics['accuracy'],
        'nan' if safe_number(test_metrics['auc']) is None else '{:.6f}'.format(test_metrics['auc'])))

    #Add the new score to the tracker which may restructured the model with PB Nodes
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_metrics['accuracy'], 
    model) 
    model.to(device)
    #If it was restructured reset the optimizer and scheduler
    if(restructured): 
        optimArgs = {'params':model.parameters(),'lr':args.lr}
        schedArgs = {'step_size':1, 'gamma': args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    metrics = {
        'validation_loss': val_metrics['loss'],
        'validation_accuracy': val_metrics['accuracy'],
        'validation_auc': val_metrics['auc'],
        'validation_precision_at_1': val_metrics['precision_at_1'],
        'test_loss': test_metrics['loss'],
        'test_accuracy': test_metrics['accuracy'],
        'test_auc': test_metrics['auc'],
        'test_precision_at_1': test_metrics['precision_at_1'],
    }
    return model, optimizer, scheduler, training_complete, metrics


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--val-split', type=float, default=0.1, metavar='N',
                        help='fraction of train set used for validation (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', 
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--width', type=float, default=1.0, metavar='M',
                        help='width multiplier')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--use-wandb', action='store_true', default=True,
                        help='enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='MNIST_PERF',
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default='PerforatedAI_IDL',
                        help='W&B entity (team) name (optional)')
    parser.add_argument('--wandb-run-name', type=str, default='',
                        help='W&B run name (optional)')
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='W&B mode')
    parser.add_argument('--wandb-api-key', type=str, default='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',
                        help='enter api key or set WANDB_API_KEY in the environment')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    total_train = len(dataset1)
    val_size = int(total_train * args.val_split)
    val_size = max(1, val_size)
    train_size = total_train - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset1, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Match PerforatedAI timing setup: CPU batch=1, GPU batch=100 for inference throughput.
    cpu_benchmark_loader = torch.utils.data.DataLoader(dataset2, batch_size=1, shuffle=False)
    gpu_benchmark_loader = torch.utils.data.DataLoader(dataset2, batch_size=100, shuffle=False)

    #Set up some global parameters for PAI code
    GPA.pc.set_testing_dendrite_capacity(False)

    model = Net(args.width)
    model = UPA.initialize_pai(model)

    model = model.to(device)
    
    #Setup the optimizer and scheduler
    GPA.pai_tracker.set_optimizer(optim.Adadelta)
    GPA.pai_tracker.set_scheduler(StepLR)
    optimArgs = {'params':model.parameters(),'lr':args.lr}
    schedArgs = {'step_size':1, 'gamma': args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    run = init_wandb(args)
    cycle_start = time.perf_counter()
    running_stats = {}
    best_validation_accuracy = float('-inf')
    best_validation_snapshot = {}


    #Run your epochs of training and testing
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete, eval_metrics = test(model, device, val_loader, test_loader, optimizer, scheduler, args)

        seconds_per_training_epoch = time.perf_counter() - epoch_start
        seconds_per_training_cycle = time.perf_counter() - cycle_start

        validation_accuracy = eval_metrics['validation_accuracy']
        validation_auc = eval_metrics['validation_auc']

        update_min_max(running_stats, 'validation_accuracy', validation_accuracy)
        update_min_max(running_stats, 'validation_auc', validation_auc)
        update_min_max(running_stats, 'test_accuracy', eval_metrics['test_accuracy'])
        update_min_max(running_stats, 'test_auc', eval_metrics['test_auc'])

        # PerforatedAI describes test scores taken at the point of highest validation score.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_validation_snapshot = {
                'test_accuracy_at_best_validation': eval_metrics['test_accuracy'],
                'test_auc_at_best_validation': eval_metrics['test_auc'],
                'validation_accuracy_best': validation_accuracy,
                'validation_auc_at_best_validation': validation_auc,
                'epoch_at_best_validation': epoch,
            }

        epoch_log = {
            'epoch': epoch,
            'perforatedai/train_accuracy': train_accuracy,
            'perforatedai/validation_accuracy': validation_accuracy,
            'perforatedai/validation_accuracy_min': running_stats.get('validation_accuracy_min'),
            'perforatedai/validation_accuracy_max': running_stats.get('validation_accuracy_max'),
            'perforatedai/validation_auc': safe_number(validation_auc),
            'perforatedai/validation_auc_min': running_stats.get('validation_auc_min'),
            'perforatedai/validation_auc_max': running_stats.get('validation_auc_max'),
            'perforatedai/test_accuracy': eval_metrics['test_accuracy'],
            'perforatedai/test_accuracy_min': running_stats.get('test_accuracy_min'),
            'perforatedai/test_accuracy_max': running_stats.get('test_accuracy_max'),
            'perforatedai/test_auc': safe_number(eval_metrics['test_auc']),
            'perforatedai/test_auc_min': running_stats.get('test_auc_min'),
            'perforatedai/test_auc_max': running_stats.get('test_auc_max'),
            'perforatedai/precision_at_1': eval_metrics['test_precision_at_1'],
            'perforatedai/seconds_per_training_epoch': seconds_per_training_epoch,
            'perforatedai/seconds_per_training_cycle': seconds_per_training_cycle,
        }

        if best_validation_snapshot:
            epoch_log['perforatedai/test_accuracy_at_best_validation'] = best_validation_snapshot['test_accuracy_at_best_validation']
            epoch_log['perforatedai/test_auc_at_best_validation'] = safe_number(best_validation_snapshot['test_auc_at_best_validation'])
            epoch_log['perforatedai/epoch_at_best_validation'] = best_validation_snapshot['epoch_at_best_validation']

        print('Epoch {} metrics: {}'.format(epoch, epoch_log))
        if run is not None:
            log_to_wandb(run, epoch_log, step=epoch)

        if(training_complete):
            break

    model.eval()
    gpu_inference_ips = float('nan')
    if torch.cuda.is_available():
        gpu_inference_ips = benchmark_inference_throughput(model, gpu_benchmark_loader, torch.device('cuda'))

    model_cpu = model.to(torch.device('cpu'))
    cpu_inference_ips = benchmark_inference_throughput(model_cpu, cpu_benchmark_loader, torch.device('cpu'))
    latency_ms = benchmark_cpu_latency_single_core_ms(model_cpu, cpu_benchmark_loader)
    param_count, flops, flops_source = compute_model_stats(model_cpu, torch.device('cpu'))

    final_metrics = {
        'perforatedai/gpu_inference_inputs_per_second': safe_number(gpu_inference_ips),
        'perforatedai/cpu_inference_inputs_per_second': safe_number(cpu_inference_ips),
        'efficientnet/num_parameters': param_count,
        'efficientnet/flops': safe_number(flops),
        'efficientnet/flops_source': flops_source,
        'efficientnet/latency_ms_per_batch': safe_number(latency_ms),
    }

    # EfficientNet-style Accuracy@Latency paired with the test score at best validation.
    if best_validation_snapshot and safe_number(latency_ms) is not None:
        final_metrics['efficientnet/accuracy_vs_flops'] = best_validation_snapshot['test_accuracy_at_best_validation']
        final_metrics['perforatedai/test_auc_at_best_validation'] = safe_number(best_validation_snapshot['test_auc_at_best_validation'])
        final_metrics['perforatedai/validation_accuracy_best'] = best_validation_snapshot['validation_accuracy_best']
        final_metrics['perforatedai/validation_auc_at_best_validation'] = safe_number(best_validation_snapshot['validation_auc_at_best_validation'])
        final_metrics['perforatedai/epoch_at_best_validation'] = best_validation_snapshot['epoch_at_best_validation']

    print('Final performance metrics: {}'.format(final_metrics))
    if run is not None:
        log_to_wandb(run, final_metrics)
        finish_wandb(run)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

#Taken from https://github.com/PerforatedAI/PerforatedAI/blob/main/Examples/baseExamples/mnist/mnist_perforatedai.py


#PerforatedAI - Accuracy, AUC (validation (min & max), test, (min & max)), Precision@1, Seconds per training epoch, seconds per training cycle, gpu inference inputs/second, cpu inference inputs/second
#EfficientNet - number of parameters, FLOPS, Accuracy @ latency

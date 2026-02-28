import sys
import time
import json
import warnings
import torch.optim as optim
from tqdm import tqdm
from models.config import args
from models.utils.mutils import *
from models.utils.data_util import *
from models.utils.util import init_logger, seed_everything
from models.utils.inits import prepare
from models.MoVE import VGAE, VGAE_MoE, GCNEncoder
from models.MoVE_runner import Runner
from models.utils.evaluation import evaluate, train_evaluate

sys.path.append("..")

warnings.simplefilter("ignore")

for runs in range(args.num_runs):
    seed_everything(runs)
    args, data = load_data(args)
    encoder = GCNEncoder(args)
    model = VGAE_MoE(args, encoder).to(args.device)
    runner = Runner(args, model, data)

    results = []
    min_epoch = args.min_epoch
    max_patience = args.patience
    patience = 0

    optimizer_env = optim.Adam(
        [p for n, p in runner.model.named_parameters() if "environment_estimator" in n],
        lr=args.env_lr, weight_decay=args.weight_decay
    )

    optimizer = optim.Adam(
        [p for n, p in runner.model.named_parameters() if "environment_estimator" not in n],
        lr=args.lr, weight_decay=args.weight_decay
    )
    log_dir = args.log_dir
    init_logger(prepare_dir(log_dir) + "log_" + args.dataset + ".txt")
    info_dict = get_arg_dict(args)

    max_auc, max_test_auc, max_train_auc = 0, 0, 0
    test_results = [0, 0, 0, 0]
    best_measure_dict = {}

    with tqdm(range(1, args.max_epoch + 1)) as bar:
        t0 = time.time()
        for epoch in bar:
            train_auc_list, val_auc_list, test_auc_list = [], [], []
            total_epoch_loss = 0.0
            num_batches = 0

            for ix, edge_index in enumerate(data['train']['edge_index_list']):

                embeddings, env_loss, align_loss = runner.model.encode(runner.x[ix], edge_index.long().to(args.device))

                if ix <= runner.len - 2:
                    auc = train_evaluate(embeddings, data['train'], ix, runner)

                    if ix < runner.len_train - 1:
                        train_auc_list.append(auc)
                    elif ix < runner.len_train + runner.len_val - 1:
                        val_auc_list.append(auc)
                    else:
                        test_auc_list.append(auc)

                    pos_edge_index = prepare(data['train'], ix + 1)[0]
                    if args.dataset == "yelp":
                        neg_edge_index = bi_negative_sampling(pos_edge_index, args.num_nodes, args.shift)
                    else:
                        neg_edge_index = negative_sampling(pos_edge_index, num_neg_samples=pos_edge_index.size(
                            1) * args.sampling_times, )

                    optimizer_env.zero_grad()
                    env_loss.backward(retain_graph=True)
                    optimizer_env.step()

                    recon_loss = model.recon_loss(embeddings, edge_index, neg_edge_index)
                    KL_loss = (1 / runner.x[ix].size(0)) * model.kl_loss()
                    loss = recon_loss + KL_loss + args.alpha * align_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_epoch_loss += loss.item()

                    current_train_auc = np.mean(train_auc_list) if train_auc_list else 0.0
                    bar.set_description(
                        f"batch_idx: {ix}, Loss: {loss.item():.4f}, AUC: {current_train_auc:.4f}, recon_loss: {recon_loss:.4f}")

            average_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0.0
            average_train_auc = np.mean(train_auc_list) if train_auc_list else 0.0
            average_val_auc = np.mean(val_auc_list) if val_auc_list else 0.0
            average_test_auc = np.mean(test_auc_list) if test_auc_list else 0.0
            if average_val_auc > max_auc:
                max_auc = average_val_auc
                max_test_auc = average_test_auc
                max_train_auc = average_train_auc
                test_results = runner.vgae_test(epoch, data["test"], runner.model.encoder.edge_decoder)
                metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(",")
                best_measure_dict = dict(zip(metrics, [max_train_auc, max_auc, max_test_auc] + test_results))
                patience = 0
                checkpoint_dir = "../checkpoint/"
                os.makedirs(checkpoint_dir, exist_ok=True)
                filepath = os.path.join(checkpoint_dir, f"MoVE{args.dataset}{runs}.pth")
                torch.save({"model_state_dict": model.state_dict()}, filepath)
            else:
                patience += 1
                if epoch > min_epoch and patience > max_patience:
                    break

            if epoch == 1 or epoch % args.log_interval == 0:
                print("Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(epoch, average_epoch_loss, time.time() - t0))
                print(
                    f"Current: Epoch:{epoch}, Train ACC:{average_train_auc:.4f}, Val ACC: {average_val_auc:.4f}, Test ACC: {average_test_auc:.4f}")
                print(
                    f"Train: Epoch:{test_results[0]}, Train ACC:{max_train_auc:.4f}, Val ACC: {max_auc:.4f}, Test ACC: {max_test_auc:.4f}")
                print(
                    f"Test: Epoch:{test_results[0]}, Train ACC:{test_results[1]:.4f}, Val ACC: {test_results[2]:.4f}, Test ACC: {test_results[3]:.4f}")
        info_dict.update(best_measure_dict)
        filename = f"info_{args.dataset}_run{runs}.json"
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, filename), "w") as f:
            json.dump(info_dict, f, indent=4)


eval_log_path = os.path.join(args.log_dir, f"eval_summary_{args.dataset}.txt")
args, data = load_data(args)
encoder = GCNEncoder(args)
model = VGAE_MoE(args, encoder).to(args.device)
runner = Runner(args, model, data)
results = evaluate(args, runner, args.num_runs)
with open(eval_log_path, "a", encoding="utf-8") as f:
    f.write(f"Dataset: {args.dataset}\n")
    f.write(f"Results: {results}\n")
    f.write("-" * 50 + "\n\n")
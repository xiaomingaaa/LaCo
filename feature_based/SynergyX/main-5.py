import os
import argparse
import os.path as osp
import time
import pandas as pd
import torch
import numpy as np

from models.model_llm import SynergyxNet
from utlis import (EarlyStopping, collect_env, load_dataloader, load_infer_dataloader, 
                   set_random_seed, train, validate, infer)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum number of epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping (default: 50)')
    parser.add_argument('--resume-from', type=str, 
                        help='the path of pretrained_model')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test or infer')               
    parser.add_argument('--omic', type=str, default='exp,mut,cn,eff,dep,met',
                        help="omics_data included in this training, separated by commas, for example: exp,mut,cn")   
    parser.add_argument('--workdir', type=str, default=os.getcwd(),
                        help='workdir of running this model')
    parser.add_argument('--celldataset', type=int, default=2,
                        help='Using which geneset to train the model(1 for 18498g, 2 for 4079g, 3 for 963g)')
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans',
                        help='cell encoder(cellTrans or cellCNNTrans)')         
    parser.add_argument('--nfold', type=str, default='0',
                        help='set index of the dataset(for example:0,1,2,indep0,blind0)')
    parser.add_argument('--dataset-split', type=str, default='v1',
                        choices=['v1', 'db', 'longtail_v1', 'longtail_db'],
                        help='which split csv set to use')
    parser.add_argument('--saved-model', type=str, 
                        help='the path of trained_model', default='./experiment/20260109_1232/0_fold_early_stop.pth')
    parser.add_argument('--infer-path', type=str, default='./MyDataset/long-tail/drugcomb_test_tail_cells.csv',
                        help="The path of the infer_data_items")
    parser.add_argument('--output-attn', type=int, default=0,
                        help="whether to output the attention matrix and cell embedding in the Infer mode(0 for not, 1 for yes)")
    return parser.parse_args()


def main():

    # pass args
    args = arg_parse()
    set_random_seed(args.seed)
    device = args.device

    # set work_dir
    work_dir = args.workdir

    # set expr_dir
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    expt_folder = osp.join('experiment/', f'{timestamp}')
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    # save environmant info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    print('\n--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('\n')
    
    
    if args.mode == 'train':
        
        nfold = [i for i in args.nfold.split(',')]
        num_repeats = 5  # 🔁 重复次数

        all_run_val_metrics = []  # list of np.array(8,)
        all_run_seeds = []        # 对应的 seed

        for repeat in range(num_repeats):
            # 当前这次重复的随机种子
            cur_seed = args.seed + repeat
            set_random_seed(cur_seed)

            print('\n' + '=' * 80)
            print(f'>>> Repeat {repeat + 1}/{num_repeats}, seed = {cur_seed}')
            print('=' * 80 + '\n')

            # 记录这一轮里，每个 fold 的验证集指标
            this_run_val_metrics = []  # (num_folds, 8)

            for k in nfold:

                model = SynergyxNet(args=args).to(device)
                # total = sum([param.nelement() for param in model.parameters()])
                # print("Number of parameter: %.2fM" % (total/1e6))
                model.init_weights()
                # criterion = torch.nn.MSELoss(reduction='mean')
                criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=args.lr,
                                             weight_decay=0.00001)
                start_epoch = 0

                # In case of unexpected interruption of model training, load the saved model and continue training
                if args.resume_from:
                    resume_path = args.resume_from
                    pretrain_dict = torch.load(resume_path)
                    model_dict = model.state_dict()
                    pretrained_dict = {k_: v for k_, v in pretrain_dict.items() if k_ in model_dict}
                    model_dict.update(pretrained_dict)
                    model.load_state_dict(model_dict)
                    start_epoch = int(osp.basename(resume_path).split('_')[0]) + 1
                    print(f'Load pre-trained parameters sucessfully! From epoch {start_epoch} to train……')

                tr_dataloader, val_dataloader, test_dataloader = load_dataloader(n_fold=k, args=args)

                start_time = time.time()
                print(f'{k}_Fold_Training is starting. Start_time:{timestamp}')

                stopper = EarlyStopping(mode='higher',
                                        metric='accuracy',
                                        patience=args.patience,
                                        n_fold=k,
                                        folder=expt_folder)
                for epoch in range(start_epoch, args.epochs):
                    train_loss, lr_list = train(model=model,
                                                criterion=criterion,
                                                opt=optimizer,
                                                dataloader=tr_dataloader,
                                                device=device,
                                                args=args)
                    val_loss, _, _, _, _, _, _, _ = validate(model=model,
                                                             criterion=criterion,
                                                             dataloader=val_dataloader,
                                                             device=device,
                                                             args=args)
                    print('Epoch %d, Train_loss %f, Valid_loss %f' % (epoch, train_loss, val_loss))

                    early_stop = stopper.step(val_loss, model)
                    if early_stop:
                        print('EarlyStopping! Finish training!')
                        break
                        
                    

                # output the performance after training
                print(f'{k}_fold training is done! Training_time:{(time.time() - start_time)/60}min')
                print('Start testing ... ')

                stopper.load_checkpoint(model)

                # 训练集指标（可选，看个大概）
                accuracy, f1, mcc, roc_auc, kappa, ap, precision, recall = validate(
                    model=model,
                    criterion=criterion,
                    dataloader=tr_dataloader,
                    device=device,
                    args=args
                )

                # 验证集指标（我们主要关心这个）
                accuracy_1, f1_1, mcc_1, roc_auc_1, kappa_1, ap_1, precision_1, recall_1 = validate(
                    model=model,
                    criterion=criterion,
                    dataloader=val_dataloader,
                    device=device,
                    args=args
                )

                print(f"Train result: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, "
                      f"ROC AUC: {roc_auc:.4f}, Kappa: {kappa:.4f}, AP: {ap:.4f}, "
                      f"Precision: {precision:.4f}, Recall: {recall:.4f}")

                print(f"Val result: Accuracy: {accuracy_1:.4f}, F1: {f1_1:.4f}, MCC: {mcc_1:.4f}, "
                      f"ROC AUC: {roc_auc_1:.4f}, Kappa: {kappa_1:.4f}, AP: {ap_1:.4f}, "
                      f"Precision: {precision_1:.4f}, Recall: {recall_1:.4f}")

                # 保存这个 fold 的验证集指标
                this_run_val_metrics.append(
                    np.array([accuracy_1, f1_1, mcc_1, roc_auc_1, kappa_1, ap_1, precision_1, recall_1])
                )

            # 这一轮 repeat 完成后，对所有 fold 的验证结果取平均
            this_run_val_metrics = np.stack(this_run_val_metrics, axis=0)  # (num_folds, 8)
            mean_over_folds = this_run_val_metrics.mean(axis=0)            # (8,)

            all_run_val_metrics.append(mean_over_folds)
            all_run_seeds.append(cur_seed)

            metric_names = ["Accuracy", "F1", "MCC", "ROC AUC", "Kappa", "AP", "Precision", "Recall"]
            print('\n>>> 当前重复 (seed = {}) 在所有 fold 上的验证集平均结果：'.format(cur_seed))
            for name, val in zip(metric_names, mean_over_folds):
                print(f"{name}: {val:.4f}")
            print('\n')

        # 所有 repeat 跑完后，对 5 组“平均验证指标”再做 mean / std
        all_run_val_metrics = np.stack(all_run_val_metrics, axis=0)  # (num_repeats, 8)
        metric_names = ["Accuracy", "F1", "MCC", "ROC AUC", "Kappa", "AP", "Precision", "Recall"]

        print('\n' + '#' * 80)
        print('>>> 每次重复（不同 seed，对所有 fold 已平均）的验证集结果：')
        for i, (seed_i, metrics_i) in enumerate(zip(all_run_seeds, all_run_val_metrics)):
            print(f'\nRun {i+1} (seed = {seed_i}):')
            for name, val in zip(metric_names, metrics_i):
                print(f"  {name}: {val:.4f}")

        final_mean = all_run_val_metrics.mean(axis=0)
        final_std = all_run_val_metrics.std(axis=0)

        print('\n' + '#' * 80)
        print('>>> 最终 5 次重复 (不同 seed) 的验证集统计结果（对 fold 已平均）：')
        for name, m, s in zip(metric_names, final_mean, final_std):
            print(f"{name}: mean = {m:.4f}, std = {s:.4f}")
        print('#' * 80 + '\n')

        print('All folds training is completed!')

    elif args.mode == 'test':
        
        print('Test mode:')
        
        device = args.device
        model = SynergyxNet(args=args).to(device)
        # load model
        saved_model = args.saved_model
        model.load_state_dict(torch.load(saved_model))
        # criterion = torch.nn.MSELoss(reduction='mean')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        k = osp.basename(saved_model).split('_')[0]
        tr_dataloader, val_dataloader, test_dataloader = load_dataloader(n_fold=k, args=args)
        accuracy, f1, mcc, roc_auc, kappa, ap, precision, recall = validate(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            device=device,
            args=args
        )
            
        print(f"Test result: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, "
              f"ROC AUC: {roc_auc:.4f}, Kappa: {kappa:.4f}, AP: {ap:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        

    elif args.mode == 'infer':
        
        print('Infer mode:')
        
        device = args.device
        model = SynergyxNet(args=args).to(device)
        # load model
        saved_model = args.saved_model
        model.load_state_dict(torch.load(saved_model))
        # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        infer_dataloader, infer_data_arr = load_infer_dataloader(args=args)
        y_pred_arr, cell_embed_arr, attn_arr = infer(
            model=model,
            dataloader=infer_dataloader,
            device=device,
            args=args
        )
        output_arr = np.concatenate((infer_data_arr, y_pred_arr), axis=1)

        print('Inferrence step is done! Saving to file……')
        output_df = pd.DataFrame(output_arr,
                                 columns=['drugA', 'drugB', 'sample_id', 'label', 'S_pred'])
        output_df.to_csv(f'experiment/{timestamp}/predict_res.csv', index=False)

        if args.output_attn:
            np.save(f'experiment/{timestamp}/cell_embed.npy', cell_embed_arr)
            np.save(f'experiment/{timestamp}/attn.npy', attn_arr)


if __name__ == '__main__':
    main()

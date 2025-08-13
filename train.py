import argparse
import torch
import os
import cpr 

log = cpr.utils.get_logger()

def main(args):
    cpr.utils.set_seed(args.seed)

    if args.emotion:
        args.data = os.path.join(
            args.data_root,
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + "_" + args.emotion + ".pkl",
        )
    else:
        args.data = os.path.join(
            args.data_root,
            args.data_dir_path,
            args.dataset,
            "data_" + args.dataset + ".pkl"
        )

    # Load data
    log.debug("Loading data from '%s'." % args.data)
    data = cpr.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = cpr.Dataset(data["train"], args)
    devset = cpr.Dataset(data["dev"], args)
    testset = cpr.Dataset(data["test"], args)

    log.debug("Building model...")
    
    model_file = os.path.join(args.data_root, "model_checkpoints", "model.pt")
    model = cpr.CPR(args).to(args.device)  
    opt = cpr.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = cpr.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Save final model
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="iemocap_4",
        choices=["iemocap", "iemocap_4", "mosei"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--data_dir_path",
        type=str,
        default="./data",
        help="Dataset directory path",
    )

    # Training parameters
    parser.add_argument(
        "--from_begin",
        action="store_true",
        default=False,
        help="Train from scratch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computing device.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        default=10,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Optimizer name.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="reduceLR",
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00025,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-8,
        help="Weight decay.",
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="Max gradient norm (set to <=0 to disable clipping).",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.5,
        help="Dropout rate.",
    )

    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=11,
        help="Past context window size. Use -1 for all past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=9,
        help="Future context window size. Use -1 for all future context.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="Hidden size.",
    )
    parser.add_argument(
        "--rnn",
        type=str,
        default="transformer",
        choices=["lstm", "transformer", "ffn"],
        help="Type of RNN encoder (or transformer).",
    )
    parser.add_argument(
        "--class_weight",
        action="store_true",
        default=False,
        help="Use class weights in NLL loss.",
    )

    # Modalities
    parser.add_argument(
        "--modalities",
        type=str,
        default="atv",
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities to use.",
    )
    parser.add_argument(
        "--gcn_conv",
        type=str,
        default="rgcn",
        choices=["rgcn"],
        help="Graph convolution layer type.",
    )

    # Emotion
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="Emotion class for datasets like mosei.",
    )

    # Additional model args
    parser.add_argument(
        "--encoder_nlayers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--graph_transformer_nheads",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--use_highway",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=24,
        help="Random seed.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=".",
        help="Data root directory.",
    )
    parser.add_argument(
        "--edge_type",
        default="temp_multi",
        choices=("temp_multi", "multi", "temp"),
        help="Edge construction type.",
    )
    parser.add_argument(
        "--use_speaker",
        action="store_true",
        default=False,
        help="Use speaker information.",
    )
    parser.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_graph_transformer",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_crossmodal",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--crossmodal_nheads",
        type=int,
        default=2,
        help="Number of attention heads in crossmodal block.",
    )
    parser.add_argument(
        "--num_crossmodal",
        type=int,
        default=2,
        help="Number of crossmodal blocks.",
    )
    parser.add_argument(
        "--self_att_nheads",
        type=int,
        default=2,
        help="Number of attention heads in self-attention block.",
    )
    parser.add_argument(
        "--num_self_att",
        type=int,
        default=3,
        help="Number of self-attention blocks.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="normalexperiment",
        help="Experiment tag (for logging or identification).",
    )

    args = parser.parse_args()

    # Embedding dimensions per dataset and modality
    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "iemocap_4": {
            "a": 100,
            "t": 768,
            "v": 512,
        },
        "mosei": {
            "a": 80,
            "t": 768,
            "v": 35,
        },
    }

    log.debug(args)

    main(args)

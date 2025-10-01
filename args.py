import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Experiment Arguments")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda:0')


    
    parser.add_argument("--task", type=str, default='Math-500')
    parser.add_argument("--range_c", type=float, default=None)
    parser.add_argument("--percentage", type=float, default=None)
    parser.add_argument("--choose", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--samples_per_task", type = int, default = 5)

    return parser.parse_args()

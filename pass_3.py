from .args import get_args


if __name__ == "__main__":
    args = get_args()
    if args.task == 'Math-500':
        from .Math500.math500 import math_pass3
        math_pass3(args)
    elif args.task == 'AIME2024':
        from .AIME2024.aime2024 import aime2024_pass3
        aime2024_pass3(args)
    elif args.task == 'AIME2025':
        from .AIME2025.aime2025 import aime2025_pass3
        aime2025_pass3(args)


        



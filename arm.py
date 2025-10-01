from .args import get_args

if __name__ == "__main__":
    args = get_args()

    if args.task == 'Math-500':
        from .Math500.math500 import math_test
        math_test(args)
    elif args.task == 'AIME2024':
        from .AIME2024.aime2024 import aime2024_test
        aime2024_test(args)
    elif args.task == 'AIME2025':
        from .AIME2025.aime2025 import aime2025_test
        aime2025_test(args)


        
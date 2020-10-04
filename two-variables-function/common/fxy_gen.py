import argparse
import numpy as np
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s generates a synthetic dataset file calling a two-variables real function on a rectangle')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.1')

    parser.add_argument('--dsout',
                        type=str,
                        dest='ds_output_filename',
                        required=True,
                        help='dataset output file (csv format)')

    parser.add_argument('--funcxy',
                        type=str,
                        dest='func_xy_body',
                        required=True,
                        help='f(x, y) body (lamba format)')

    parser.add_argument('--xbegin',
                        type=float,
                        dest='range_xbegin',
                        required=False,
                        default=-5.0,
                        help='begin x range (default:-5.0)')

    parser.add_argument('--xend',
                        type=float,
                        dest='range_xend',
                        required=False,
                        default=+5.0,
                        help='end x range (default:+5.0)')

    parser.add_argument('--ybegin',
                        type=float,
                        dest='range_ybegin',
                        required=False,
                        default=-5.0,
                        help='begin y range (default:-5.0)')

    parser.add_argument('--yend',
                        type=float,
                        dest='range_yend',
                        required=False,
                        default=+5.0,
                        help='end y range (default:+5.0)')

    parser.add_argument('--xstep',
                        type=float,
                        dest='range_xstep',
                        required=False,
                        default=0.01,
                        help='step range of x (default: 0.01)')

    parser.add_argument('--ystep',
                        type=float,
                        dest='range_ystep',
                        required=False,
                        default=0.01,
                        help='step range of y (default: 0.01)')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    x_values = np.arange(args.range_xbegin, args.range_xend, args.range_xstep, dtype=float)
    y_values = np.arange(args.range_ybegin, args.range_yend, args.range_ystep, dtype=float)
    func_xy = eval('lambda x, y: ' + args.func_xy_body)
    csv_ds_output_file = open(args.ds_output_filename, 'w')
    with csv_ds_output_file:
        writer = csv.writer(csv_ds_output_file, delimiter=',')
        writer.writerow(['x', 'y', 'z'])
        for i in range(0, x_values.size):
            for j in range(0, y_values.size):
                writer.writerow([x_values[i], y_values[j], func_xy(x_values[i], y_values[j])])
    print("Generated two-variables scalar function synthetic dataset file '%s'" % args.ds_output_filename)

    print("#### Terminated %s ####" % os.path.basename(__file__));

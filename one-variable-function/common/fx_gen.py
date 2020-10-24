import argparse
import numpy as np
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s generates a synthetic dataset file calling a one-variable real function in an interval')

    parser.add_argument('--version', action='version', version='%(prog)s 1.1.0')

    parser.add_argument('--dsout',
                        type=str,
                        dest='ds_output_filename',
                        required=True,
                        help='dataset output file (csv format)')

    parser.add_argument('--funcx',
                        type=str,
                        dest='func_x_body',
                        required=True,
                        help='f(x) body (lamba format)')

    parser.add_argument('--xbegin',
                        type=float,
                        dest='range_begin',
                        required=False,
                        default=-5.0,
                        help='begin range (default:-5.0)')

    parser.add_argument('--xend',
                        type=float,
                        dest='range_end',
                        required=False,
                        default=+5.0,
                        help='end range (default:+5.0)')

    parser.add_argument('--xstep',
                        type=float,
                        dest='range_step',
                        required=False,
                        default=0.01,
                        help='step range (default: 0.01)')

    parser.add_argument('--noise',
                        type=str,
                        dest='noise_body',
                        required=False,
                        help='noise(sz) body (lamba format)')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    x_values = np.arange(args.range_begin, args.range_end, args.range_step, dtype=float)
    if args.noise_body:
        noise_sz = eval('lambda sz: ' + args.noise_body)
        noise =  noise_sz(len(x_values))
    else:
        noise = [0] * len(x_values)

    func_x = eval('lambda x: ' + args.func_x_body)
    csv_ds_output_file = open(args.ds_output_filename, 'w')
    with csv_ds_output_file:
        writer = csv.writer(csv_ds_output_file, delimiter=',')
        writer.writerow(['x', 'y'])
        for i in range(0, x_values.size):
            writer.writerow([x_values[i], func_x(x_values[i])  + noise[i]])
    print("Generated one-variable scalar function synthetic dataset file '%s'" % args.ds_output_filename)

    print("#### Terminated %s ####" % os.path.basename(__file__));

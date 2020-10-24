import argparse
import numpy as np
import csv
import os

def generate_noise(noise_body, t_values):
    if noise_body:
        noise_sz = eval('lambda sz: ' + noise_body)
        noise =  noise_sz(len(t_values))
    else:
        noise = [0] * len(t_values)
    return noise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s generates a synthetic dataset file that contains the points of a parametric curve on plane calling a couple of one-variable real functions in an interval1')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    parser.add_argument('--dsout',
                        type=str,
                        dest='ds_output_filename',
                        required=True,
                        help='dataset output file (csv format)')

    parser.add_argument('--funcxt',
                        type=str,
                        dest='funcx_t_body',
                        required=True,
                        help='x=x(t) body (lamba format)')

    parser.add_argument('--funcyt',
                        type=str,
                        dest='funcy_t_body',
                        required=True,
                        help='y=y(t) body (lamba format)')

    parser.add_argument('--tbegin',
                        type=float,
                        dest='range_begin',
                        required=False,
                        default=-5.0,
                        help='begin range (default:-5.0)')

    parser.add_argument('--tend',
                        type=float,
                        dest='range_end',
                        required=False,
                        default=+5.0,
                        help='end range (default:+5.0)')

    parser.add_argument('--tstep',
                        type=float,
                        dest='range_step',
                        required=False,
                        default=0.01,
                        help='step range (default: 0.01)')

    parser.add_argument('--noisex',
                        type=str,
                        dest='noisex_body',
                        required=False,
                        help='noise(sz) body (lamba format) on x')

    parser.add_argument('--noisey',
                        type=str,
                        dest='noisey_body',
                        required=False,
                        help='noise(sz) body (lamba format) on y')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    t_values = np.arange(args.range_begin, args.range_end, args.range_step, dtype=float)
    noisex = generate_noise(args.noisex_body, t_values)
    noisey = generate_noise(args.noisey_body, t_values)

    funcx_t = eval('lambda t: ' + args.funcx_t_body)
    funcy_t = eval('lambda t: ' + args.funcy_t_body)
    csv_ds_output_file = open(args.ds_output_filename, 'w')
    with csv_ds_output_file:
        writer = csv.writer(csv_ds_output_file, delimiter=',')
        writer.writerow(['t', 'x', 'y'])
        for i in range(0, t_values.size):
            writer.writerow([t_values[i], funcx_t(t_values[i]) + noisex[i], funcy_t(t_values[i]) + noisey[i]])
    print("Generated one-variable scalar function synthetic dataset file '%s'" % args.ds_output_filename)

    print("#### Terminated %s ####" % os.path.basename(__file__));

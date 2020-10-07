import argparse, textwrap
import csv
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
    %(prog)s shows two alongside x/y/z scatter graphs: \n
    the blue one is the surface of dataset,\n
    the red one is the surface of prediction
    '''))

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.1')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format)')

    parser.add_argument('--prediction',
                        type=str,
                        dest='prediction_data_filename',
                        required=True,
                        help='prediction data file (csv format)')

    parser.add_argument('--title',
                        type=str,
                        dest='figure_title',
                        required=False,
                        default='',
                        help='if present, it set the title of chart')

    parser.add_argument('--xlabel',
                        type=str,
                        dest='x_axis_label',
                        required=False,
                        default='',
                        help='label of x axis')

    parser.add_argument('--ylabel',
                        type=str,
                        dest='y_axis_label',
                        required=False,
                        default='',
                        help='label of y axis')

    parser.add_argument('--zlabel',
                        type=str,
                        dest='z_axis_label',
                        required=False,
                        default='',
                        help='label of z axis')

    parser.add_argument('--labelfontsize',
                        type=int,
                        dest='label_font_size',
                        required=False,
                        default=9,
                        help='label font size')

    parser.add_argument('--width',
                        type=float,
                        dest='width',
                        required=False,
                        default=9.60,
                        help='width of animated git (in inch)')

    parser.add_argument('--height',
                        type=float,
                        dest='height',
                        required=False,
                        default=5.40,
                        help='height of animated git (in inch)')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the chart is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    plt.rcParams.update({'font.size': args.label_font_size})
    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel(args.x_axis_label, fontdict={'size': args.label_font_size})
    ax1.set_ylabel(args.y_axis_label, fontdict={'size': args.label_font_size})
    ax1.set_zlabel(args.z_axis_label, fontdict={'size': args.label_font_size})

    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            ax1.scatter(float(row[0]), float(row[1]), float(row[2]), color='blue', s=1, marker='.')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel(args.x_axis_label, fontdict={'size': args.label_font_size})
    ax2.set_ylabel(args.y_axis_label, fontdict={'size': args.label_font_size})
    ax2.set_zlabel(args.z_axis_label, fontdict={'size': args.label_font_size})
    with open(args.prediction_data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            ax2.scatter(float(row[0]), float(row[1]), float(row[2]), color='red', s=2, marker='.')

    fig.suptitle(args.figure_title, fontdict={'size': args.label_font_size, 'color': 'orange'})
    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));

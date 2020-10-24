import argparse, textwrap
import csv
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=textwrap.dedent('''\
    %(prog)s shows two overlapped x/y scatter graphs:
    the blue one is the input dataset of parametric curve,
    the red one is the prediction of parametric curve
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
                        help='width of whole figure (in inch)')

    parser.add_argument('--height',
                        type=float,
                        dest='height',
                        required=False,
                        default=5.40,
                        help='height of whole figure (in inch)')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the figure is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    plt.rcParams.update({'font.size': args.label_font_size})
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    ax.set_title(args.figure_title, fontdict={'size': args.label_font_size, 'color': 'orange'})
    ax.set_xlabel(args.x_axis_label, fontdict={'size': args.label_font_size})
    ax.set_ylabel(args.y_axis_label, fontdict={'size': args.label_font_size})

    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            plt.scatter(float(row[1]), float(row[2]), color='blue', s=1, marker='.')

    with open(args.prediction_data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            plt.scatter(float(row[1]), float(row[2]), color='red', s=2, marker='.')

    plt.title(args.figure_title);
    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()

    print("#### Terminated %s ####" % os.path.basename(__file__));

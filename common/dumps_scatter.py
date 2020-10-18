import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(data_filename, colorname):
    plt.clf()
    y_name = os.path.splitext(data_filename)[0]
    data_values = np.genfromtxt(os.path.join(args.dump_path, data_filename), delimiter=',')
    plt.title(data_filename)
    plt.xlabel('epochs')
    plt.ylabel(y_name)
    plt.plot(data_values, color=colorname)

    if args.save_figure_directory:
        if not os.path.exists(args.save_figure_directory):
            os.makedirs(args.save_figure_directory)
        fig_file_path = os.path.join(args.save_figure_directory, y_name) + '.png'
        plt.savefig(fig_file_path)
        print("Saved file '%s'" % fig_file_path);
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='%(prog)s shows the loss and metric graphs with data generated by any fitting program with argument --dumpout')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.1')

    parser.add_argument('--dump',
                        type=str,
                        dest='dump_path',
                        required=True,
                        help='Dump directory (generated by any fitting/training programs of this suite that support --dumpout argument)')

    parser.add_argument('--savefigdir',
                        type=str,
                        dest='save_figure_directory',
                        required=False,
                        default='',
                        help='If present, the charts are saved on files in savefig_dir folder instead to be shown on screen')

    args = parser.parse_args()

    print("#### Started %s ####" % os.path.basename(__file__));

    loss_filenames = [mfn for mfn in os.listdir(args.dump_path) if mfn.startswith('loss_')]
    if len(loss_filenames) > 0:
        loss_filename = loss_filenames[0]
        plot_graph(loss_filename, 'm')

    metric_filenames = [mfn for mfn in os.listdir(args.dump_path) if mfn.startswith('metric_')]
    for metric_filename in metric_filenames:
        plot_graph(metric_filename, 'tab:orange')

    val_filenames = [mfn for mfn in os.listdir(args.dump_path) if mfn.startswith('val_')]
    for val_filename in val_filenames:
        plot_graph(val_filename, 'tab:blue')

    print("#### Terminated %s ####" % os.path.basename(__file__));

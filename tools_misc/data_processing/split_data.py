import os
import argparse
import random
import math


# def fil_path(string):
#     if os.path.isfile(string):
#         return string
#     else:
#         raise NotAFileError(string)


def split_data(file_path, seed=42, training_ratio=0.7):
  with open(file_path) as f:
    next(f)
    examples_list = f.read().splitlines()
    random.seed(seed)
    random.shuffle(examples_list)

    # Divide the data into training and validation set
    num_train = int(math.ceil(training_ratio * len(examples_list)))
    print(
        'Total number of original images in training set: {}'.format(num_train))
    print('Total number of original images in validation set: {}'
          .format(len(examples_list) - num_train))
    train_set = examples_list[:num_train]
    eval_set = examples_list[num_train:]
    return train_set, eval_set



#os.path.join(dataset_dir, filename)

def main():
    parser = argparse.ArgumentParser(description='Splitting into train and eval')
    parser.add_argument('--file_path', type = str,
            default = '/scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/metadata.csv',
            help = 'file path of total train csv')
    parser.add_argument('--new_train_csv', type = str, default=None,
            help = 'file path of split train csv')
    parser.add_argument('--new_eval_csv', type = str, default = None,
            help = 'file path of split eval csv')
    args = parser.parse_args()
    file_path = args.file_path
    new_train_csv = args.new_train_csv
    new_eval_csv = args.new_eval_csv

    train_set, eval_set = split_data(file_path)
    #print(train_set)
    outfile = open(new_train_csv, 'w')
    outfile.write('\n'.join(train_set))
    outfile.close()
    outfile = open(new_eval_csv, 'w')
    outfile.write('\n'.join(eval_set))
    outfile.close()


if __name__ == '__main__':
  main()

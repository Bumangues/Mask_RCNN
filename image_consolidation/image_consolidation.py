import os
import csv
import sys
import getopt
import shutil

training_csv = open('training/labels.csv', 'a')
validation_csv = open('validation/labels.csv', 'a')

def main(argv):
    training_csv.write('img_id,bbox_id,x_min,x_max,y_min,y_max,label_l1,label_l2,label'+'\n')
    validation_csv.write('img_id,bbox_id,x_min,x_max,y_min,y_max,label_l1,label_l2,label'+'\n')

    csvs_dir = 'csv_labels/'
    dst_dir = 'training/'
    try:
        opts, args = getopt.getopt(argv, "hc:", ["csvs="])
    except getopt.GetoptError:
        print('image_consolidation.py -t <training_split>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('image_consolidation.py -c <csvs>')
            sys.exit()
        elif opt in ("-c", "--csvs"):
            csvs_dir = arg

    csvs = os.listdir(csvs_dir)
    count = 0
    rows = []

    old_img_id = ''
    current_img_id = ''

    for c in csvs:
        with open(csvs_dir + c, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            fields = next(csv_reader)

            old_img_id = ''
            for row in csv_reader:
                old_img_id = current_img_id
                current_img_id = str(row[0])

                if old_img_id == '':
                    rows.append(row)
                elif old_img_id == current_img_id:
                    rows.append(row)
                else:
                    count = insert_and_move(count, rows, old_img_id)
                    rows = []

    if len(rows) > 0:
        insert_and_move(count, rows, rows[0][0])

    training_csv.close()
    validation_csv.close()


def insert_and_move(count, rows, old_img_id):
    dst_dir = 'training/'
    if count % 5 < 3:
        dst_dir = 'training/'
        for r in rows:
            training_csv.write(', '.join(r) + '\n')
    if count % 5 >= 3:
        dst_dir = 'validation/'
        for r in rows:
            validation_csv.write(', '.join(r) + '\n')
    shutil.copy('all_images/' + str(old_img_id) + '.jpg', dst_dir + "images/")
    return count + 1


if __name__ == "__main__":
    main(sys.argv[1:])

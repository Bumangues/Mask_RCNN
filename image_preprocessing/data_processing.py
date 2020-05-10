from PIL import Image, ImageDraw
import os
import csv
import shutil

new_csv = open('labels.csv', 'a')

dst_dir = 'processed_images/'
l = os.listdir(dst_dir) # dir is your directory path
number_files = len(l)
print('Files Processed: ', number_files)

with open('labels/foid_labels_bbox_v012.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    src_img = 'images/{image}.jpg'
    dst_img = dst_dir + '{image}.jpg'

    last_image = ''
    image = ''
    boxes = []
    rows = []

    count = 0

    for row in csvreader:
        try:
            Image.open(src_img.format(image=row[0]))
        except FileNotFoundError:
            continue

        if row[7] == 'HUMAN':
            last_image = image
            image = row[0]
            box = row[2:6]

            if last_image == '':
                boxes.append(box)
                rows.append(row)
            elif image == last_image:
                boxes.append(box)
                rows.append(row)
            else:
                try:
                    print('SRC: ', src_img.format(image=last_image))
                    print('DST: ', dst_img.format(image=last_image))
                    shutil.copy(src_img.format(image=last_image), dst_dir)
                except FileNotFoundError:
                    continue
                img = Image.open(src_img.format(image=last_image))
                draw = ImageDraw.Draw(img)
                num = 0
                for b in boxes:
                    x_min = int(b[0])
                    x_max = int(b[1])
                    y_min = int(b[2])
                    y_max = int(b[3])

                    draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                    draw.text((x_min, y_min), str(num), stroke_width=10, stroke_fill='red')
                    num += 1
                img.show()

                dangerous_input = input("Type skip to ignore, exit to exit the program or list which boxes are dangerous (space separated): ")
                dangerous_labels = dangerous_input.split(' ')

                if 'exit' in dangerous_labels:
                    os.remove(dst_img.format(image=last_image))
                    break
                elif 's' in dangerous_labels:
                    boxes = []
                    rows = []
                    boxes.append(box)
                    rows.append(row)

                    os.remove(dst_img.format(image=last_image))
                    os.remove(src_img.format(image=last_image))
                    continue

                for i in range(len(boxes)):
                    if str(i) in dangerous_labels:
                        rows[i].append('1')
                        new_csv.write(', '.join(val for val in rows[i])+'\n')
                    else:
                        rows[i].append('0')
                        new_csv.write(', '.join(val for val in rows[i])+'\n')
                count += 1
                print('Files Processed: ', count)
                print('Total Files Processed: ', count + number_files)

                boxes = []
                rows = []
                boxes.append(box)
                rows.append(row)
                os.remove(src_img.format(image=last_image))

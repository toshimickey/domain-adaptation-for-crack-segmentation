# dataフォルダ内のデータを前処理する
# input:
#   - data/Conglomerate
#   - data/Chundata
# output:
#   - data/Train
#   - data/Test
#   - data/original_split_resized
#   - data/teacher_split_resized

import os
import shutil
from PIL import Image

# パスの設定
conglomerate_train_dir = 'data/Conglomerate/Train'
conglomerate_test_dir = 'data/Conglomerate/Test'
data_dir = 'data'
chundata_original_dir = 'data/Chundata/original'
chundata_teacher_dir = 'data/Chundata/teacher'
output_dir_original = 'data/original_split_resized'
output_dir_teacher = 'data/teacher_split_resized'

# 必要なフォルダを作成
os.makedirs(output_dir_original, exist_ok=True)
os.makedirs(output_dir_teacher, exist_ok=True)

shutil.move(conglomerate_train_dir, data_dir)
shutil.move(conglomerate_test_dir, data_dir)


# 画像を分割して保存する関数
def split_image(image_path, save_dir, rows=6, cols=9, tile_size=(576, 576), extension="jpg"):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path)
    tile_width, tile_height = tile_size

    for row in range(rows):
        for col in range(cols):
            left, upper = col * tile_width, row * tile_height
            right, lower = left + tile_width, upper + tile_height
            tile = image.crop((left, upper, right, lower))
            tile_name = f"{image_name}_{row+1}_{col+1}.{extension}"
            tile.save(os.path.join(save_dir, tile_name))
    
    print(f"{image_path} を分割して {save_dir} に保存しました。")

# 指定フォルダ内の画像を分割して保存
def process_images(input_folder, output_folder, file_extension="jpg", tile_extension="jpg"):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(f".{file_extension}"):
            image_path = os.path.join(input_folder, filename)
            split_image(image_path, output_folder, extension=tile_extension)

# originalフォルダのjpgファイルを分割して保存
process_images(chundata_original_dir, output_dir_original, file_extension="jpg", tile_extension="jpg")

# teacherフォルダのbmpファイルを分割して保存
process_images(chundata_teacher_dir, output_dir_teacher, file_extension="bmp", tile_extension="bmp")

print("すべての処理が完了しました。")

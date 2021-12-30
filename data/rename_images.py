import copy
import os

if __name__ == '__main__':
    main_input_path = '/home/grzegorz/projects/museum/tagging-tool/tagging_tool/static/images/portrait'
    bad_signs = ['&', '#', ';', '?']
    all_image_names = os.listdir(main_input_path)

    for image_name in all_image_names:
        is_unwanted_sing = any([x in bad_signs for x in image_name])

        if is_unwanted_sing:
            new_image_name = copy.deepcopy(image_name)
            for bad_sign in bad_signs:
                new_image_name = new_image_name.replace(bad_sign, '')
            print(f'renamed {image_name} to {new_image_name}')
            os.rename(
                os.path.join(main_input_path, image_name),
                os.path.join(main_input_path, new_image_name)
            )

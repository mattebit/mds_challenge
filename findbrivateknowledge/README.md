# FindBrivateKnowledge CTM
## Description
This repository contains the following scripts:
- `findbrivateknowledge_attack.py` this file contains all the attacks that can be used;
- `findbrivateknowledge_demo.py` this file contains a demo main function which performs embedding, attack and detection on a sample image;
- `findbrivateknowledge_detection.py` this file contains the detection and extraction functions to be used on the attacked images;
- `findbrivateknowledge_embedding.py` this file contains the embedding functions to be used on the images;
- `findbrivateknowledge_roc_curve.py` this file contains the ROC curve generation function used to calculate the threshold value for the detection function;

## Attack methodology
The attacking methodology is based on the following steps:
1. prepare the list of attacks to be performed as a list of lists, where each list contains dictionaries with the attack function and the attack parameters (refer to the `findbrivateknowledge_attack.py` file for the list of available attacks and a sample list);
2. use the `apply_attacks` function to apply the attacks to the image, passing each element of the attacks list, the image, and the detection function as arguments;
   - To improve performances multiprocessing is used in the `findbrivateknowledge_attack.py` file, where a `ProcessPoolExecutor` is used to launch the functions and gather the results asynchronously.
3. the `apply_attacks` function will perform the attacks on the image and return the resulting image, the result of the detection function on the attacked image, the attacked image's WPSNR and the attack parameters used;

## Demo usage
To run the demo, simply run the following command:
```bash
python3 findbrivateknowledge_demo.py
```
This script embeds the watermark in the `lena_grey.bmp` image and then performs two attacks (one successful and one unsuccessful) on the image. Thanks to the `apply_attacks` function being used, the detection function can be passed as an argument to the attack function, which will then be used to detect the watermark in the attacked image and return the result.

## Single script usage
To use the scripts individually, you can simply run them using python. Each script has a main function that can be used to test them. Some scripts require some files to be present in the same directory, such as the `lena_grey.bmp` image or the results of another script, it is recommended to run the demo script first to generate the necessary files.
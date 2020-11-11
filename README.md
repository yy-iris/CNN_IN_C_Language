# CNN_IN_C_Language
This repository is written in C and realize the feedward calculation of CNN.

**config:**
python: 3.6  
keras: 2.3.1  
tensorflos: 1.14.0  
I run the c code in VS2017 and the IDE will raise error "VS error c4996: 'fopen': This function or variable may be unsafe". You could add `; _CRT_SECURE_NO_WARNINGS` (notify the `;`) in 'project properties >> c/c++ >> preprocessor >> preprocessor definition'.  

**0. get the model**  
First of all, you should prepare your own `.h5` file, or you can use the `train_model()` in `produce_model.py` I prepare for you if you just want a simple try.

**1. process the input data**  
rename the input file into "input.csv" or "pic.jpg" and run `data_prep.py` in `pyFile` fold and transform your input data into `.txt` format. It can process both `.jpg` file and '.csv' file and the generated file will be stored in the `input_images_txt` fold.  
Note: your data should be the same as you input in python, such as normalization.

**2. generate C file and extract weights from `.h5`**  
Then you should run `generator.py` which includes two functions:  
1) `generate()` will generate a file named `model.c`. It includes `main()` function which you should run after all preparations.  
2) `extract_weights()` could extract weights from `.h5` file and store them in `model_weights_txt` fold, which are used for feedward calculation in `model.c`.  
Note: if you use the loss function defined by yourself, you should set the `custom_objections` in `keras.models.load_model()`.  

After step 2 and step 3, you will get the `model.c`, `input_images_txt` fold and `model_weights_txt` fold, then you should remove them into the right place in `cFile` fold. In this repository, I build the project in VS2017. I put `model.c` in `cFile`, while `input_images_txt` fold and `model_weights_txt` fold in `cFile/conv/conv` (`conv` is the name of project).  

**3. get the final result**  
Final step, run `model.c` in `cFile` fold and get the final result.  

Note:  
1) there is a function named `show_matrix()` in `cnn_inference.c` through which you could see the output in each layer or save it into `.txt` file.  
2) Besides `train_model()` function, there are other two functions in `produce_model.py`. `predict_mode()` could get the final result through python, and you could compare it with result get from `model.c`. Under normal circumstances, they should be the same. `forward_check()` could get the output of each layer. It is convenient for debuging.  

Actually, this reposity is modified from [Can Yalniz](https://github.com/canyalniz/CNN-Inference-Didactic). I add some functions and fix some bugs.  
1) add the softmax function in `cnn_inference.c`.  
2) make the channel of input into a variable. It is a const value (3) in original repository, which is not general. For example, the image in mnist only has one channel.  
3) fix the bugs in `Conv()`, `MaxPool()` and `FlattenD()`.  
- calculation of padding and output dimension in `Conv()` and `MaxPool()` are not right.  
- direction of flatten in `FlattenD()` is not right.  





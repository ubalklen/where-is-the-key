# Where is the key?

Machine Learning applied to the problem of finding Wally's keys.

## Introduction

*Where's Wally* is a heavily illustrated puzzle book series created by Martin Handford where you have to find a little guy in very chaotic scenarios. This seemed to me a good problem to be solved by a computer using Machine Learning (ML) techniques.

After a web search returned a [first page](https://www.google.com/search?q=where's+wally+waldo+machine+learning) full of results of people that had already done that, I changed the task to an arguably more difficult one (for humans, at least): finding Wally's *keys* (one of the many side goals of the books).

As an example, here is one of the books scenarios:

![Scenario 2-1](http://i.imgur.com/ybsexIE.jpg)

And here is the Wally's key in that scenario:

![Scenario 2-2 key](https://github.com/ubalklen/where-is-the-key/blob/master/data/3-slices/2-2-k-2816-1979.jpg/70-49-k.jpg)

(Go ahead and try yourself. If you give up, here is the [answer](https://github.com/ubalklen/where-is-the-key/blob/master/data/1-original-images/where_s_waldo_now____book_2___scene_2__by_where_is_waldo_wally_d7naw60.jpg).)

## The data

All the data I used to train this ML model is based on [Where-is-Waldo-Wally](https://www.deviantart.com/where-is-waldo-wally/gallery/) DevianArt Gallery. They had scanned all the scenes from books #2 to #5 and added convenient marks pointing to the answers. All the originals images can be found in [data/1-original-images](https://github.com/ubalklen/where-is-the-key/tree/master/data/1-original-images) folder.

From those images, I cutted out the answers panel ([data/2-no-panels](https://github.com/ubalklen/where-is-the-key/tree/master/data/2-no-panels) folder). I then chopped the resulting images into 40x40 sided slices ([data/3-slices](https://github.com/ubalklen/where-is-the-key/tree/master/data/3-slices) folder and subfolders).

The images filenames are in *00-00-k.jpg* format, where the numbers are the coordinates of the slice and the *-k* means there is a key in the image.

In the 3-slices folder, there is also a subfolder named [augmented](https://github.com/ubalklen/where-is-the-key/tree/master/data/3-slices/augmented) with rotated versions of the keys to address the issue of unbalanced data (there are many more images with no keys than otherwise). If you want to know more, Bharath Raj wrote a nice [article](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced) about data augmentation.

Unfortunately, my model didn't take much advantage of all those images because I had had to discard most of them before training the model. My computer couldn't handle the load.

## The model

The model have been constructed with evilsocket's [Ergo](https://github.com/evilsocket/ergo) framework, which makes easier to build models on [Keras](https://keras.io/).

I tried to use the same model evilsocket had used to demonstrate Ergo (https://www.evilsocket.net/2018/11/22/Presenting-project-Ergo-how-to-build-an-airplane-detector-for-satellite-imagery-with-Deep-Learning/), a vanilla Convolutional Neural Network (CNN) for airplane detection in satellite imagery. Unfortunately, results have not been very good.

I ended up making the network deeper and things improved. Here is the final model:

![Where is the key model](https://github.com/ubalklen/where-is-the-key/blob/master/model.png)


## Results

I was able to get the following results:
```
Training --------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     23789
           1       1.00      1.00      1.00     12040

   micro avg       1.00      1.00      1.00     35829
   macro avg       1.00      1.00      1.00     35829
weighted avg       1.00      1.00      1.00     35829


confusion matrix:

[[23787     2]
 [    3 12037]]

Validation ------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5030
           1       0.99      1.00      1.00      2647

   micro avg       1.00      1.00      1.00      7677
   macro avg       1.00      1.00      1.00      7677
weighted avg       1.00      1.00      1.00      7677


confusion matrix:

[[5011   19]
 [   1 2646]]

Test ------------------------------------------------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5085
           1       0.99      1.00      1.00      2592

   micro avg       1.00      1.00      1.00      7677
   macro avg       1.00      1.00      1.00      7677
weighted avg       1.00      1.00      1.00      7677


confusion matrix:

[[5060   25]
[ 0 2592]]
```

## Final notes

Thanks to all the dev whose work allowed me to create this project.
Thank you, Gaby (*personal cheerleader*).

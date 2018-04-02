

** Machine Learning NanoDegree (Final Project)
			License Plate Recognition

there are 3 models:
	1- Train License Classifier(yes/no)
	2- Train Character Classifier(yes/no)
	3- Train Character Recognition(A->Z,a->z,0->10)

Data consist of:
	1- Images With License Plate with Label = 1 ,and Othe Images not contain License with Label = 0
	2- Images with Charcters with Label = 1, and other Images(space,white space,image contain part of two letters) with Label = 0
	3- images for character from [A->Z,a->z,0->10] with labels of there indexs different images for the same character with different font for training 

Preporcessing:
	* Load the images 
	* resize the images to the same size(100,100)
	* adding labels for every different class
	* randomize the data to overcame any Pattern in the data
	* split the data into training and testing
	* split training data into training and validation


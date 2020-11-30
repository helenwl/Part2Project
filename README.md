# Part2Project
Creating an MoS2 defect detection neural network

Workflow of the project:

1. generate coordinates of an MoS2 lattice (with random line defects in it- defectivelatticegeneration.py)
2. simulate HRTEM images of specific defective lattices (example_defective_HRTEM_MULTEM.m)
3. each defective lattice is used to generate several images with MULTEM (by varying the input parameters- found in example_HRTEM_MULTEM file.m)
4. each image produced is 'noised' so that simulated HRTEM images mimic experimental data
5. each image has a corresponding 'mask' which represents the ground truth for that simulated image- this is key to supervised learning
6. the collection of images produced via steps 1-5 are fed into a neural network
7. apply the 80:20 principle, so save 20% of my training data to test my network and assess it's accuracy. Be careful not to overfit.
8. evaluate the success/effectiveness of the model

Things to consider:
How accurate is the simulation? What assumptions does it make in order to model? Is it reliable?
How can we quantify the success of a neural network? 
How do multiline defects behave? Can we track defects (object detection) to prove that vacancies agglomerate to form line defects?
Where is the my model (made in step 1) unrealistic?
How can we optimise the network without overfitting to the model?
Is there an optimum number of layers? How much training data is needed to produce a given peformance?

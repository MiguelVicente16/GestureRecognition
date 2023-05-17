README file
-----------
These data are described in the following paper:

Huang, Jaiswal & Rai (2019). “Gesture-based system for next generation natural and
intuitive interfaces”. Artificial Intelligence for Engineering Design, Analysis and Manu-
facturing, 33 (1), pp. 54-68.

Only the Domain 1 and 4 will be investigated (Domain 1 first because simpler). Therefore, the data corresponding to each of these two domains are contained in two different directories.

Each .csv file contains the tracking of exactly one 3-dimensional (3D) hand gesture signal (x,y,z) along time (t). Therefore, each line of the file contains 4 numbers, (x,y,z,t).

The file name is made of (1) the Subject's index/number (0 to 9, the person on which the gesture is recorded), (2) the 3D gesture's type (0 to 9 for Domain 1 and three-dimensional figures, or symbols, for Domain 2, like Cone, Cylinder, ...), and (3) the Repetition number (1 to 10) because each 3D gesture is repeated/recorded 10 times for each Subject.

Thus, for each Domain, there are 10 Subjects x 10 3D figures x 10 Repetitions = 1000 signals in total.

We invite the reader to look at the paper of Huang et al. (2019), available on Moodle, for more information.
# MontionMingle

**Developer Names:** Qi Shu, Xunzhou Ye, Anhao Jiao, Kehao Huang, Qianlin Chen

**Date of project start:** 2023-09-16

**This project is the McMaster University Software Engineering 2023-2024 Capstone Project.**

MontionMingle is an innovative online Tai Chi learning platform. It provides a real-time 
video streaming platform as a web application for both Tai-chi instructors and practitioners. 
The instructors are able to start a training session and stream their video captured by 
the webcam. The practitioners are able to join the training session and watch the 
live video from the instructor. 
    
Additionally, during a training session, all users are able to turn on real-time annotations 
rendered on the streaming video. In the current system, we implemented three types of annotations, 
the skeleton, footwork and semantic segmentation. They are generated through the machine learning 
pipeline running on a server. These annotations are aimed to help practitioners understand and 
mimic the movement of the instructor, and therefore significantly improve their learning outcomes. 
Besides that, all of the annotations are user-configurable. Every single practitioner are able to 
select their preferred annotation to watch, and they are able to seamlessly switch between 
annotations anytime they want. 

**Instructor cllient interface:**

![image](https://github.com/InfiniView-AI/MotionMingle/assets/77683292/36ace217-4d0d-4609-bac1-b703b90fb426)

**Skeleton annotation:** Human skeleton joints to better indicate the Tai Chi instructor’s movement. 
According to a previous research done by one of our supervisor’s research groups. The skeleton annotation 
is one of the most popular annotations by the targeted audience group.

![image](https://github.com/InfiniView-AI/MotionMingle/assets/77683292/d4c40ba5-20ea-4ac1-8c24-f912e7f4b961)

**Footwork annotation:** One of the most popular annotations according to the research results. Tai Chi 
emphasizes whole-body movement. However, when watching an online Tai Chi video stream, it is sometimes 
hard to tell the center of mass/support foot of the instructor when the practitioner. This annotation 
helps users to see the Tai Chi footwork part more clearly.

![image](https://github.com/InfiniView-AI/MotionMingle/assets/77683292/a8dac3a8-07e5-4513-a2a5-6d668746fc84)

**Semantic segmentation:** According to AOA, human vision peaks between age 19 to 40, 
after that, elders might start to have trouble distinguishing the background and the instructor. 
Therefore, this segmentation annotation dims the instructor’s background, and human eyes will just 
naturally focus on the instructor’s movement to enhance their learning experience.

![image](https://github.com/InfiniView-AI/MotionMingle/assets/77683292/34d1a3e1-9d7a-43cd-8805-0abf6d8936f7)


**Folders and files structure:**

**docs** - Documentation for the project

**refs** - Reference material used for the project, including papers

**src**- Source code

**test** - Test cases


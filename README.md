# Welcome to Section Seeker!

![Section-Seeker-Logo](assets/reflect_connect.png)

#### **This notebook and package has been adapted from ThinkOnward's [Reflection Connection Challenge](https://thinkonward.com/app/c/challenges/reflection-connection), which ran in late 2023. The SectionSeeker can be used to train a SiameseNN to identify similar sections to the one a user inputs. This can be extremely useful for seismic interpreters looking for an analog section or basin.**


#### Background

Siamese Neural Networks (SNN) have shown great skill at one-shot learning collections of various images.  This challenge asks you to train an algorithm to find similar-looking images of seismic data within a larger corpus using a limited training for eight categories.  Your solution will need to match as many different features using these data.  This challenge is experimental, so we are keen to see how different participants utilize this framework to build a solution.

To non-geophysicists, seismic images are mysterious: lots of black-and-white squiggly lines stacked on one another.  However, with more experience, different features in the seismic can be identified.  These features represent common geology structures: a river channel, a salt pan, or a fault.  Recognizing seismic features is no different from when a medical technician recognizes the difference between a heart valve or major artery on an echocardiogram.  A geoscientist combines all these features into a hypothesis about how the Earth developed in the survey area.  An algorithm that can identify parts of a seismic image will enable geoscientists to build more robust hypotheses and spend more time integrating other pieces of information into a comprehensive model of the Earth.

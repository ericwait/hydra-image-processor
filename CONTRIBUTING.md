# Extending HIP is easy
Hydra Image Processor (HIP) has been written to expedite the addition of functionality. All of the machinery needed to distribute data and iterate over neighborhoods is included. Adding a new function is as easy as copying a template file and replacing the requisite lines of code. This paradigm is intended to encourage research and development of operations that would otherwise be intractable without hardware acceleration. 

# How to contribute
Contribution is as easy as a pull request on GitHub. Following the guidelines will ensure requests are not rejected for minor issues. Once your code conforms to the guidelines, please submit your requests [here](https://github.com/ericwait/hydra-image-processor/pulls). 

Alternatively, you can contribute by detailing your needs on this [forum](https://www.hydraimageprocessor.com/forum/request-functionality). HIP was also designed to be a practical library that meets the needs of microscopist. I really enjoy when theory and application meet. By explaining in detail (examples are always helpful as well) what your needs are, the community will be able to find novel ways accomplish your goals. 

# Guidelines
## Code is not correct until it is clean

From the very beginning HIP was written to be clean, consistent, and as clear as possible. The use of object oriented programming and templates have made this project quickly extensible as well as maintainable. Each portion of code, down to the lowest operation, should be as "_glanceable_" as possible. Meaning that use of spacing, letter case, and naming scheme should be as information dense as possible. It all boils down to, make code that assists the reader in their understanding of what it is intended to accomplish.

## Main points to follow

1. Format, format, format. Make it look clean and consistent with existing code.
1. Do not duplicate functionality. Use existing functions/classes when possible. Consider extending existing functionality before creating new.
1. Use templates to ensure code maintainability. 
1. Mimic what as already been done. If code looks inconsistent or "out of place." Correct it or bring it to the community's attention.
1. _**Try crazy things**_! HIP was built to quickly get operations onto GPU hardware. Use this opportunity to create things that would not otherwise be tractable. 

### These guidelines are intended for GitHub pull requests. Do not let them be a barrier to experimentation. Have FUN!

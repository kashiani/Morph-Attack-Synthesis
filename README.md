
# Face Morph Attack Synthesis

The **Face Morph Attack Synthesis** project provides a comprehensive framework designed for synthesizing high-quality facial morphs. It aims to explore vulnerabilities in face recognition systems (FRSs), particularly under scenarios like automatic border control (ABC) gates, where the integrity of biometric verification is crucial. The project supports three primary morphing methods: **StyleGAN-Based Morphing**, **OpenCV-Based Morphing**, and **FaceMorpher-Based Morphing**. This toolkit is essential for researchers and developers working in biometrics, computer vision, and adversarial attack, aiming to enhance the security measures against face morphing attacks.

---

## Features

- **StyleGAN-Based Morphing**:
  - Leverages state-of-the-art GANs for realistic facial morph generation with fine-grained control.
- **OpenCV-Based Morphing**:
  - In landmark-based morph generation, OpenCV is utilized to detect critical facial landmarks, ensuring precise alignment of faces before morphing.
- **FaceMorpher-Based Morphing**:
  - Simplifies face morphing through automated landmark detection, Delaunay triangulation, and pixel-wise blending for efficient and effective transformations.
- **Customizable Inversion Techniques for StyleGAN-Based Morphing**:
  - **Image2StyleGAN (I2S)** for detailed inversion using StyleGAN.
  - **Landmark-Based Inversion** for morphing based on facial landmarks. Focuses on integrating facial landmarks into the morphing process to ensure that the morphed images maintain a high degree of authenticity and can effectively challenge FRSs.
  - Offers options for adjusting and warping landmarks before inversion, refining the morph to closely match or blend the features of the subjects involved.
- **Flexible Output Directory Structure**:
  - Organizes results by morphing method, inversion type, warping status, and image pair.

---

## Project Structure

```plaintext
Face-Morph-Attack-Synthesis/
│
├── src/
│   ├── inversion/                         # Inversion methods for morphing
│   │   ├── image_inverter.py              # Image2StyleGAN inversion  +  Warping + I2S inversion
│   │   ├── landmark_inverter.py           # Landmark-based inversion +  Warping + Landmark-based inversion
│   │   ├── Image2StyleGAN.py              #  
│   │   ├── Image2StyleGAN_convex_hull.py  # 
│   ├── warping/                 # Warping utilities
│   │   ├── warper.py            # Core warping logic
│   ├── morphing/                # Morphing methods
│   │   ├── latent_morpher.py    # Performs latent morphing for given coefficients and saves the results
│   │   ├── opencv_morph.py      # Morphing using OpenCV 
│   │   ├── facemorpher.py       # Morphing using FaceMorpher
│   ├── utils/                   # Utilities
│   │   ├── file_utils.py        # File management utilities
│   │   ├── image_utils.py       # Image processing utilities
│   │   ├── align_image.py       # Image alignment utility
│   ├── main.py                  # Main entry point for the project
│
├── examples/                    # Example inputs and outputs
│   ├── input_images/            # Input images for morphing
│   ├── morph_pairs.txt          # Input pairs for morphing
│
├── requirements.txt             # Dependencies for the project
├── README.md                    # Project documentation
└── LICENSE                      # License for the project
```
 
---


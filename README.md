# EdgeFX - Edge Detection for GTA San Andreas

Advanced Canny edge detection plugin that adds cinematic black outlines to objects/textures' edges in GTA San Andreas, enhancing visual depth and creating a distinctive comic style.

## What It Does

EdgeFX analyzes the game's graphics in real-time and draws black edges around objects and on textures, creating a comic book visual effect. The edges automatically adapt to object distance, contrast, and scene complexity for optimal visual quality.

## Key Features

### 🎨 **Visual Enhancement**
- **Cinematic Outlines**: Adds black edges around objects and on textures for dramatic visual impact
- **Distance-Based Adaptation**: Edges automatically adjust based on how far objects are from the camera
- **Contrast Sensitivity**: Edge thickness varies based on local image contrast for natural appearance
- **Real-Time Processing**: Analyzes and processes graphics in real-time without performance impact

### 🔧 **Technical Capabilities**
- **Canny Edge Detection**: Professional-grade algorithm for precise edge detection
- **Dynamic Thresholding**: Automatically adjusts sensitivity based on scene complexity
- **Depth Buffer Integration**: Uses GTA SA's depth information for accurate distance calculations
- **6 Blending Modes**: Different visual styles from pure black to bloom-compatible effects
- **Separate Alpha Controls**: Independent transparency for low and high contrast edges

### ⚡ **Performance & Compatibility**
- **Minimal Impact**: Designed for smooth gameplay with minimal framerate loss
- **GTA SA Optimized**: Specifically tuned for GTA San Andreas graphics and performance
- **Bloom Compatible**: Special modes for use with bloom/lighting mods
- **Works with SkyGFX**: Compatible with other visual mods and enhancements

## Installation

1. **Requirements**: GTA San Andreas (v1.0 US), .asi loader, DirectX 9 graphics card
2. **Compilation**: Uses Plugin SDK with SkyGFX reference examples
3. **Installation**: Place compiled `.asi` file in GTA SA directory
4. **Configuration**: Edit `EdgeFX.ini` (created automatically on first run)

## Visual Impact

EdgeFX transforms GTA San Andreas into a visually striking experience with:
- **Enhanced Depth Perception**: Objects and textures stand out clearly from backgrounds
- **Cinematic Atmosphere**: Creates a movie-like visual style
- **Artistic Appeal**: Adds a unique artistic comic touch to the game
- **Improved Clarity**: Makes objects/structures and textures more defined

## Technical Details

- **Performance**: Minimal framerate impact with optimized shaders
- **Optimization**: Highest quality settings pre-configured
- **Compatibility**: Works with bloom mods and other visual enhancements

## Credits

- **Zeneric**: EdgeFX Plugin creator/author
- **Plugin SDK**: Development framework
- **SkyGFX**: Post processing and stuff implementation reference
- **Cursor AI**: Development assistance 
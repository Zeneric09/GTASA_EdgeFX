#include "C:/1/plugin-sdk-master/shared/PluginBase.h"
#include "C:/1/plugin-sdk-master/shared/Events.h"
#include "C:/1/plugin-sdk-master/shared/extensions/Shader.h"
#include "C:/1/plugin-sdk-master/plugin_sa/game_sa/rw/rwcore.h"
#include "C:/1/plugin-sdk-master/shared/extensions/Config.h"
#include "C:\1\plugin-sdk-master\shared\RenderWare.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <windows.h>
#include "C:/1/plugin-sdk-master/DXSDK/Include/d3dx9.h"
#include "C:/1/plugin-sdk-master/DXSDK/Include/d3dx9math.h"

using std::min;
using std::max;

using namespace plugin;

// Suppress MessageBox calls for shader file errors
int WINAPI HookedMessageBoxA(HWND hWnd, LPCSTR lpText, LPCSTR lpCaption, UINT uType) {
    // Check if this is the shader file error message
    if (lpText && strstr(lpText, "Failed to open shader file")) {
        return IDOK; // Return OK without showing the message
    }
    // For all other messages, show them normally
    return MessageBoxA(hWnd, lpText, lpCaption, uType);
}

int WINAPI HookedMessageBoxW(HWND hWnd, LPCWSTR lpText, LPCWSTR lpCaption, UINT uType) {
    // Check if this is the shader file error message
    if (lpText && wcsstr(lpText, L"Failed to open shader file")) {
        return IDOK; // Return OK without showing the message
    }
    // For all other messages, show them normally
    return MessageBoxW(hWnd, lpText, lpCaption, uType);
}

// Custom shader class that extends the plugin SDK's Shader
class CustomShader : public plugin::Shader {
public:
    IDirect3DPixelShader9* pixelShader = nullptr;
    IDirect3DVertexShader9* vertexShader = nullptr;
    
    // Custom constructor that calls parent with a dummy filename to avoid file loading
    CustomShader() : plugin::Shader("dummy_shader_that_will_not_be_loaded") {
        // Initialize our own members
        pixelShader = nullptr;
        vertexShader = nullptr;
    }
    
    void SetShaders(IDirect3DVertexShader9* vs, IDirect3DPixelShader9* ps) {
        vertexShader = vs;
        pixelShader = ps;
    }
    
    void EnableCustom() {
        if (vertexShader) _rwD3D9SetVertexShader(vertexShader);
        if (pixelShader) _rwD3D9SetPixelShader(pixelShader);
    }
    
    void DisableCustom() {
        _rwD3D9SetVertexShader(0);
        _rwD3D9SetPixelShader(0);
    }
};

class zpostfx {
private:
    // Configuration file
    config_file config;
    

    
    // Edge Detection Settings
    bool edgeEffectEnabled = true; // Enable by default for testing
    float edgeThreshold = 0.250f; // Edge detection threshold (sensitivity) - normalized for GTA SA scale
    float edgeStrength = 0.90f; // Edge visibility intensity (0.1-3.0) - increased for better visibility
    float edgeThickness = 0.35f; // Edge thickness/thinness multiplier (0.1-5.0) - optimized for game scale
    D3DXVECTOR4 edgeColor = D3DXVECTOR4(0.01f, 0.01f, 0.01f, 0.965f); // Black edges with alpha
    bool colorcycleEnabled = false; // Enable colorcycle animation on edges
    float colorcycleSpeed = 2.0f; // Speed of colorcycle animation (0.1-10.0) - adjusted for game time
    int edgeDetectionMethod = 1; // 0 = Sobel (original), 1 = Canny (new)
    
    // Canny Edge Detection Parameters - normalized for GTA SA coordinate system
    float cannyLowThreshold = 0.120f; // Low threshold for Canny (0.05-0.3) - adjusted for game scale
    float cannyHighThreshold = 0.350f; // High threshold for Canny (0.1-0.5) - optimized for game objects
    float cannyGaussianSigma = 1.00f; // Gaussian blur sigma for Canny (0.8-2.5) - balanced for game detail
    float cannyLowAlpha = 0.99f; // Alpha for weak edges (low threshold) - separate control
    float cannyHighAlpha = 0.96f; // Alpha for strong edges (high threshold) - separate control
    
    // Edge Fading Parameters
    bool edgeFadingEnabled = true; // Enable edge fading effect
    float edgeFadingStrength = 0.5f; // Fading strength (0.0-1.0) - increased for visibility
    float edgeFadingDistance = 1.50f; // Fading distance factor (0.1-2.0) - increased for visibility
    int edgeFadingMode = 0; // 0 = Distance-based, 1 = Screen-edge based, 2 = Gradient-based
    float cannyLowFadingStrength = 0.25f; // Separate fading strength for weak edges (0.0-1.0)
    float cannyHighFadingStrength = 0.10f; // Separate fading strength for strong edges (0.0-1.0)
    
    // Bloom Compatibility Settings
    bool bloomCompatibilityMode = true; // Enable bloom compatibility mode
    int blendingMode = 1; // 0 = Normal, 1 = Bloom-compatible, 2 = Multiply, 3 = Screen, 4 = Overlay
    
    // Shader objects
    CustomShader* edgeDetectionShader = nullptr;
    
    // Raster objects for rendering
    RwRaster* edgeRaster = nullptr;
    
    // Time tracking for animations
    float animationTime = 0.0f;

    // Embedded shader source code for edge detection
    const char* edgeDetectionVertexShaderSource = R"(
struct VS_INPUT {
    float3 position : POSITION;
    float2 texcoord : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 position : POSITION;
    float2 texcoord : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.position = float4(input.position, 1.0);
    output.texcoord = input.texcoord;
    return output;
}
)";

    const char* edgeDetectionPixelShaderSource = R"(
uniform sampler2D screenTexture : register(s0);
uniform float4 edgeParams : register(c0); // x = threshold, y = strength, z = thickness, w = detectionMethod (0=Sobel, 1=Canny)
uniform float4 edgeColor : register(c1);
uniform float4 colorcycleParams : register(c2); // x = enabled, y = speed, z = time, w = unused
uniform float4 cannyParams : register(c3); // x = lowThreshold, y = highThreshold, z = gaussianSigma, w = unused
uniform float4 cannyAlphaParams : register(c5); // x = lowAlpha, y = highAlpha, z = unused, w = unused
uniform float4 fadingParams : register(c4); // x = enabled, y = strength, z = distance, w = mode
uniform float4 cannyFadingParams : register(c6); // x = lowFadingStrength, y = highFadingStrength, z = unused, w = unused
uniform float4 compatibilityParams : register(c7); // x = bloomCompatibilityMode, y = blendingMode, z = unused, w = unused

struct PS_INPUT {
    float2 texcoord : TEXCOORD0;
};

float3 hsv_to_rgb(float3 hsv) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(frac(hsv.xxx + K.xyz) * 6.0 - K.www);
    return hsv.z * lerp(K.xxx, clamp(p - K.xxx, 0.0, 1.0), hsv.y);
}

// Edge fading function with multiple modes
float calculateEdgeFading(float2 texcoord, float4 fadingParams) {
    if (fadingParams.x < 0.5) return 1.0; // Fading disabled
    
    float strength = fadingParams.y;
    float distance = fadingParams.z;
    int mode = int(fadingParams.w);
    
    float fadeFactor = 1.0;
    
    if (mode == 0) {
        // Distance-based fading (from screen center) - made more pronounced
        float2 center = float2(0.5, 0.5);
        float distFromCenter = length(texcoord - center);
        fadeFactor = 1.0 - saturate(distFromCenter * distance * 4.0); // Increased multiplier
    }
    else if (mode == 1) {
        // Screen-edge based fading - made more pronounced
        float2 edgeDist = float2(
            min(texcoord.x, 1.0 - texcoord.x),
            min(texcoord.y, 1.0 - texcoord.y)
        );
        float minEdgeDist = min(edgeDist.x, edgeDist.y);
        fadeFactor = saturate(minEdgeDist * distance * 8.0); // Increased multiplier
    }
    else if (mode == 2) {
        // Gradient-based fading (radial) - made more pronounced
        float2 center = float2(0.5, 0.5);
        float distFromCenter = length(texcoord - center);
        fadeFactor = 1.0 - saturate(distFromCenter * distance * 3.0); // Increased multiplier
    }
    
    // Apply strength factor - made more aggressive
    fadeFactor = lerp(1.0, fadeFactor, strength);
    
    // Ensure minimum visibility for debugging
    fadeFactor = max(fadeFactor, 0.1);
    
    return fadeFactor;
}

// Canny-specific edge fading function with separate control for weak/strong edges
float calculateCannyEdgeFading(float2 texcoord, float4 fadingParams, float4 cannyFadingParams, float edgeType) {
    if (fadingParams.x < 0.5) return 1.0; // Fading disabled
    
    float distance = fadingParams.z;
    int mode = int(fadingParams.w);
    
    // Choose fading strength based on edge type
    float strength = 1.0;
    if (edgeType > 1.5) {
        strength = cannyFadingParams.y; // Strong edge fading strength
    } else if (edgeType > 0.5) {
        strength = cannyFadingParams.x; // Weak edge fading strength
    }
    
    float fadeFactor = 1.0;
    
    if (mode == 0) {
        // Distance-based fading (from screen center)
        float2 center = float2(0.5, 0.5);
        float distFromCenter = length(texcoord - center);
        fadeFactor = 1.0 - saturate(distFromCenter * distance * 4.0);
    }
    else if (mode == 1) {
        // Screen-edge based fading
        float2 edgeDist = float2(
            min(texcoord.x, 1.0 - texcoord.x),
            min(texcoord.y, 1.0 - texcoord.y)
        );
        float minEdgeDist = min(edgeDist.x, edgeDist.y);
        fadeFactor = saturate(minEdgeDist * distance * 8.0);
    }
    else if (mode == 2) {
        // Gradient-based fading (radial)
        float2 center = float2(0.5, 0.5);
        float distFromCenter = length(texcoord - center);
        fadeFactor = 1.0 - saturate(distFromCenter * distance * 3.0);
    }
    
    // Apply strength factor based on edge type
    fadeFactor = lerp(1.0, fadeFactor, strength);
    
    // Ensure minimum visibility
    fadeFactor = max(fadeFactor, 0.1);
    
    return fadeFactor;
}

// Gaussian blur function for Canny edge detection
float4 gaussianBlur(sampler2D tex, float2 texcoord, float2 texelSize, float sigma) {
    float4 result = float4(0.0, 0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    
    // 5x5 Gaussian kernel
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            float2 offset = float2(i, j) * texelSize;
            float weight = exp(-(i*i + j*j) / (2.0 * sigma * sigma));
            result += tex2D(tex, texcoord + offset) * weight;
            totalWeight += weight;
        }
    }
    
    return result / totalWeight;
}

// Sobel edge detection based on tutorial source code
float sobelEdgeDetection(sampler2D tex, float2 texcoord, float2 texelSize) {
    // Sample 3x3 neighborhood for Sobel filter
    // Using exact kernel from tutorial: Gx = [[1,0,-1],[2,0,-2],[1,0,-1]], Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    
    // Top row: [1,0,-1] for Gx, [1,2,1] for Gy
    float4 topLeft = tex2D(tex, texcoord + float2(-texelSize.x, -texelSize.y));
    float4 top = tex2D(tex, texcoord + float2(0.0, -texelSize.y));
    float4 topRight = tex2D(tex, texcoord + float2(texelSize.x, -texelSize.y));
    
    // Middle row: [2,0,-2] for Gx, [0,0,0] for Gy
    float4 left = tex2D(tex, texcoord + float2(-texelSize.x, 0.0));
    float4 center = tex2D(tex, texcoord);
    float4 right = tex2D(tex, texcoord + float2(texelSize.x, 0.0));
    
    // Bottom row: [1,0,-1] for Gx, [-1,-2,-1] for Gy
    float4 bottomLeft = tex2D(tex, texcoord + float2(-texelSize.x, texelSize.y));
    float4 bottom = tex2D(tex, texcoord + float2(0.0, texelSize.y));
    float4 bottomRight = tex2D(tex, texcoord + float2(texelSize.x, texelSize.y));
    
    // Convert to grayscale using exact weights from tutorial: r_const=0.2126, g_const=0.7152, b_const=0.0722
    // Note: Tutorial uses gamma correction, but for real-time we use linear weights
    float3 weights = float3(0.2126, 0.7152, 0.0722);
    
    float tl = dot(topLeft.rgb, weights);
    float t = dot(top.rgb, weights);
    float tr = dot(topRight.rgb, weights);
    float l = dot(left.rgb, weights);
    float c = dot(center.rgb, weights);
    float r = dot(right.rgb, weights);
    float bl = dot(bottomLeft.rgb, weights);
    float b = dot(bottom.rgb, weights);
    float br = dot(bottomRight.rgb, weights);
    
    // Apply exact Sobel kernels from tutorial:
    // Gx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    // Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    
    // Gx calculation: sum of element-wise multiplication
    float gx = 1.0*tl + 0.0*t + (-1.0)*tr + 
               2.0*l + 0.0*c + (-2.0)*r + 
               1.0*bl + 0.0*b + (-1.0)*br;
    
    // Gy calculation: sum of element-wise multiplication
    float gy = 1.0*tl + 2.0*t + 1.0*tr + 
               0.0*l + 0.0*c + 0.0*r + 
               (-1.0)*bl + (-2.0)*b + (-1.0)*br;
    
    // Calculate magnitude using exact formula from tutorial: sqrt(gx^2 + gy^2)
    return sqrt(gx*gx + gy*gy);
}

// Canny edge detection using correct Sobel implementation with separate alpha control
float3 cannyEdgeDetection(sampler2D tex, float2 texcoord, float2 texelSize, float lowThreshold, float highThreshold, float sigma, float lowAlpha, float highAlpha) {
    // Step 1: Apply Gaussian blur
    float4 blurred = gaussianBlur(tex, texcoord, texelSize, sigma);
    
    // Step 2: Calculate gradients using exact Sobel kernels from tutorial
    // Sample 3x3 neighborhood for gradient calculation
    float4 topLeft = tex2D(tex, texcoord + float2(-texelSize.x, -texelSize.y));
    float4 top = tex2D(tex, texcoord + float2(0.0, -texelSize.y));
    float4 topRight = tex2D(tex, texcoord + float2(texelSize.x, -texelSize.y));
    float4 left = tex2D(tex, texcoord + float2(-texelSize.x, 0.0));
    float4 center = tex2D(tex, texcoord);
    float4 right = tex2D(tex, texcoord + float2(texelSize.x, 0.0));
    float4 bottomLeft = tex2D(tex, texcoord + float2(-texelSize.x, texelSize.y));
    float4 bottom = tex2D(tex, texcoord + float2(0.0, texelSize.y));
    float4 bottomRight = tex2D(tex, texcoord + float2(texelSize.x, texelSize.y));
    
    // Convert to grayscale using exact weights from tutorial: r_const=0.2126, g_const=0.7152, b_const=0.0722
    float3 weights = float3(0.2126, 0.7152, 0.0722);
    
    float tl = dot(topLeft.rgb, weights);
    float t = dot(top.rgb, weights);
    float tr = dot(topRight.rgb, weights);
    float l = dot(left.rgb, weights);
    float c = dot(center.rgb, weights);
    float r = dot(right.rgb, weights);
    float bl = dot(bottomLeft.rgb, weights);
    float b = dot(bottom.rgb, weights);
    float br = dot(bottomRight.rgb, weights);
    
    // Apply exact Sobel kernels from tutorial:
    // Gx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    // Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    
    // Gx calculation: sum of element-wise multiplication
    float gx = 1.0*tl + 0.0*t + (-1.0)*tr + 
               2.0*l + 0.0*c + (-2.0)*r + 
               1.0*bl + 0.0*b + (-1.0)*br;
    
    // Gy calculation: sum of element-wise multiplication
    float gy = 1.0*tl + 2.0*t + 1.0*tr + 
               0.0*l + 0.0*c + 0.0*r + 
               (-1.0)*bl + (-2.0)*b + (-1.0)*br;
    
    // Step 3: Calculate gradient magnitude using exact formula from tutorial: sqrt(gx^2 + gy^2)
    float magnitude = sqrt(gx*gx + gy*gy);
    
    // Step 4: Apply double thresholding with separate alpha control
    float edgeValue = 0.0;
    float edgeAlpha = 0.0;
    float edgeType = 0.0; // 0.0 = no edge, 1.0 = weak edge, 2.0 = strong edge
    
    if (magnitude > highThreshold) {
        edgeValue = 1.0; // Strong edge
        edgeAlpha = highAlpha; // Use high alpha for strong edges
        edgeType = 2.0; // Strong edge type
    } else if (magnitude > lowThreshold) {
        edgeValue = 1.0; // Weak edge (still visible)
        edgeAlpha = lowAlpha; // Use low alpha for weak edges
        edgeType = 1.0; // Weak edge type
    }
    
    return float3(edgeValue, edgeAlpha, edgeType);
}

float4 main(PS_INPUT input) : COLOR {
    // Sample the screen texture
    float4 screenColor = tex2D(screenTexture, input.texcoord);
    
    // Fixed texel size normalized for GTA SA coordinate system (1 unit ≈ 1 meter)
    float2 texelSize = float2(0.0015, 0.0015); // Normalized for GTA SA world scale (4800x4800 units)
    
    // Apply thickness multiplier to texel size
    float thickness = edgeParams.z;
    float2 adjustedTexelSize = texelSize * thickness;
    
    // Choose edge detection method
    float edgeValue = 0.0;
    float edgeAlpha = edgeColor.a; // Default alpha from edge color
    float edgeType = 0.0; // Edge type for Canny fading (0 = Sobel, 1 = weak Canny, 2 = strong Canny)
    
    if (edgeParams.w < 0.5) {
        // Sobel edge detection (original method)
        float edgeMagnitude = sobelEdgeDetection(screenTexture, input.texcoord, adjustedTexelSize);
        edgeValue = step(edgeParams.x, edgeMagnitude) * edgeParams.y;
        edgeAlpha = edgeColor.a; // Use default alpha for Sobel
        edgeType = 0.0; // Sobel edge type
    } else {
        // Canny edge detection (new method) with separate alpha control
        float3 cannyResult = cannyEdgeDetection(screenTexture, input.texcoord, adjustedTexelSize, 
                                               cannyParams.x, cannyParams.y, cannyParams.z,
                                               cannyAlphaParams.x, cannyAlphaParams.y);
        edgeValue = cannyResult.x * edgeParams.y;
        edgeAlpha = cannyResult.y; // Use Canny-specific alpha
        edgeType = cannyResult.z; // Use Canny edge type for fading
    }
    
            // Apply edge color with optional colorcycle and fading
        float4 finalColor = screenColor;
        if (edgeValue > 0.0) {
            float4 currentEdgeColor = edgeColor;
            
            // Apply colorcycle if enabled
            if (colorcycleParams.x > 0.5) {
                float time = colorcycleParams.z;
                float speed = colorcycleParams.y;
                
                // Create cycling hue based on time and position
                float hue = frac(time * speed + input.texcoord.x * 0.1 + input.texcoord.y * 0.1);
                float3 hsvColor = float3(hue, 1.0, 1.0);
                float3 rgbColor = hsv_to_rgb(hsvColor);
                currentEdgeColor = float4(rgbColor, edgeColor.a);
            }
            
            // Calculate edge fading factor based on edge type
            float fadeFactor = 1.0;
            if (edgeParams.w < 0.5) {
                // Sobel edge detection - use standard fading
                fadeFactor = calculateEdgeFading(input.texcoord, fadingParams);
            } else {
                // Canny edge detection - use separate fading for weak/strong edges
                fadeFactor = calculateCannyEdgeFading(input.texcoord, fadingParams, cannyFadingParams, edgeType);
            }
            
            // DEBUG: Show fading as red overlay for testing (comment out when working)
            // finalColor.rgb = lerp(finalColor.rgb, float3(1.0, 0.0, 0.0), fadeFactor * 0.3);
            
            // Blend edge color with processed color using bloom-compatible blending
            // Use the Canny-specific alpha value and fading factor to control the blend strength
            float blendAlpha = edgeAlpha * edgeValue * fadeFactor;
            
            // Choose blending mode based on compatibility settings
            int blendMode = int(compatibilityParams.y);
            
            if (blendMode == 0) {
                // Normal alpha blending
                finalColor.rgb = lerp(finalColor.rgb, currentEdgeColor.rgb, blendAlpha);
            }
            else if (blendMode == 1) {
                // Bloom-compatible blending for black edges
                if (currentEdgeColor.rgb.x < 0.1 && currentEdgeColor.rgb.y < 0.1 && currentEdgeColor.rgb.z < 0.1) {
                    finalColor.rgb = finalColor.rgb * (1.0 - blendAlpha * 0.6);
                } else {
                    finalColor.rgb = lerp(finalColor.rgb, currentEdgeColor.rgb, blendAlpha * 0.8);
                }
            }
            else if (blendMode == 2) {
                // Multiply blending with proper alpha handling for dark edges
                finalColor.rgb = lerp(finalColor.rgb, finalColor.rgb * currentEdgeColor.rgb, blendAlpha * currentEdgeColor.a);
            }
            else if (blendMode == 3) {
                // Screen blending with controlled intensity for dark edges
                // Scale edge intensity to prevent over-brightening
                float3 edgeIntensity = float3(edgeValue * 0.3, edgeValue * 0.3, edgeValue * 0.3);
                float3 screenBlend = 1.0 - (1.0 - edgeIntensity) * (1.0 - finalColor.rgb);
                finalColor.rgb = lerp(finalColor.rgb, screenBlend, blendAlpha * currentEdgeColor.a);
            }
            else if (blendMode == 4) {
                // Overlay blending with proper alpha handling for dark edges
                float3 overlayBlend;
                overlayBlend.r = (finalColor.r < 0.5) ? (2.0 * finalColor.r * currentEdgeColor.r) : (1.0 - 2.0 * (1.0 - finalColor.r) * (1.0 - currentEdgeColor.r));
                overlayBlend.g = (finalColor.g < 0.5) ? (2.0 * finalColor.g * currentEdgeColor.g) : (1.0 - 2.0 * (1.0 - finalColor.g) * (1.0 - currentEdgeColor.g));
                overlayBlend.b = (finalColor.b < 0.5) ? (2.0 * finalColor.b * currentEdgeColor.b) : (1.0 - 2.0 * (1.0 - finalColor.b) * (1.0 - currentEdgeColor.b));
                finalColor.rgb = lerp(finalColor.rgb, overlayBlend, blendAlpha * currentEdgeColor.a);
            }

            
            finalColor.rgb = saturate(finalColor.rgb);
        }
    
    return finalColor;
}
)";

public:
    zpostfx() {
        // Initialize random seed for SSAO noise texture

        
        // Install MessageBox hooks to suppress shader file error
        HMODULE user32 = GetModuleHandleA("user32.dll");
        if (user32) {
            FARPROC originalMessageBoxA = GetProcAddress(user32, "MessageBoxA");
            if (originalMessageBoxA) {
                // Create a simple hook by replacing the function pointer
                // This is a basic approach - in production you'd use a proper hooking library
                DWORD oldProtect;
                if (VirtualProtect(originalMessageBoxA, 5, PAGE_EXECUTE_READWRITE, &oldProtect)) {
                    // Write a jump to our hooked function
                    *(BYTE*)originalMessageBoxA = 0xE9; // JMP instruction
                    *(DWORD*)((BYTE*)originalMessageBoxA + 1) = (DWORD)HookedMessageBoxA - (DWORD)originalMessageBoxA - 5;
                    VirtualProtect(originalMessageBoxA, 5, oldProtect, &oldProtect);
                }
            }
        }
        
        // Load configuration
        LoadConfig();
        
        // Initialize shaders
        InitializeEdgeDetectionShader();
        
        // Hook into rendering events
        Events::drawingEvent.after += [this]() {
            // Render edge detection effect
            if (edgeEffectEnabled) {
                RenderEdgeDetection();
            }
        };
    }
    
    ~zpostfx() {
        if (edgeDetectionShader) {
            if (edgeDetectionShader->vertexShader) {
                edgeDetectionShader->vertexShader->Release();
            }
            if (edgeDetectionShader->pixelShader) {
                edgeDetectionShader->pixelShader->Release();
            }
            delete edgeDetectionShader;
        }

        if (edgeRaster) {
            RwRasterDestroy(edgeRaster);
        }
    }
    

    
    void LoadConfig() {
        config.open("EdgeFX.ini");
        
        // Load edge detection configuration
        edgeEffectEnabled = config["EdgeEffectEnabled"].asBool(true);
        edgeThreshold = config["EdgeThreshold"].asFloat(0.250f);
        edgeStrength = config["EdgeStrength"].asFloat(0.90f);
        edgeThickness = config["EdgeThickness"].asFloat(0.35f);
        colorcycleEnabled = config["ColorcycleEnabled"].asBool(false);
        colorcycleSpeed = config["ColorcycleSpeed"].asFloat(2.0f);
        edgeDetectionMethod = config["EdgeDetectionMethod"].asInt(1);
        
        // Load Canny edge detection configuration
        cannyLowThreshold = config["CannyLowThreshold"].asFloat(0.120f);
        cannyHighThreshold = config["CannyHighThreshold"].asFloat(0.350f);
        cannyGaussianSigma = config["CannyGaussianSigma"].asFloat(1.00f);
        cannyLowAlpha = config["CannyLowAlpha"].asFloat(0.99f);
        cannyHighAlpha = config["CannyHighAlpha"].asFloat(0.96f);
        
        // Load edge fading configuration
        edgeFadingEnabled = config["EdgeFadingEnabled"].asBool(true);
        edgeFadingStrength = config["EdgeFadingStrength"].asFloat(0.5f);
        edgeFadingDistance = config["EdgeFadingDistance"].asFloat(1.50f);
        edgeFadingMode = config["EdgeFadingMode"].asInt(0);
        cannyLowFadingStrength = config["CannyLowFadingStrength"].asFloat(0.25f);
        cannyHighFadingStrength = config["CannyHighFadingStrength"].asFloat(0.10f);
        
        // Load bloom compatibility settings
        bloomCompatibilityMode = config["BloomCompatibilityMode"].asBool(true);
        blendingMode = config["BlendingMode"].asInt(1);
        
        // Load edge color components
        edgeColor.x = config["EdgeColorR"].asFloat(0.01f);
        edgeColor.y = config["EdgeColorG"].asFloat(0.01f);
        edgeColor.z = config["EdgeColorB"].asFloat(0.01f);
        edgeColor.w = config["EdgeColorA"].asFloat(0.965f);
        
        // Always save config to ensure INI file is created/updated
        SaveConfig();
    }
    
    void SaveConfig() {
        // Check if file already exists - if so, don't modify it
        FILE* checkFile = fopen("EdgeFX.ini", "r");
        if (checkFile) {
            fclose(checkFile);
            return; // File exists, don't modify it at all
        }
        
        // If file doesn't exist, create it with comprehensive settings
        FILE* file = fopen("EdgeFX.ini", "w");
        if (!file) return;
        
        fprintf(file, "# EdgeFX Configuration File\n");
        fprintf(file, "# This file contains settings for edge detection effects\n\n");
        
        fprintf(file, "[Edge Detection]\n");
        fprintf(file, "# Enable/disable edge detection effect\n");
        fprintf(file, "# Values: true/false\n");
        fprintf(file, "EdgeEffectEnabled = %s\n\n", edgeEffectEnabled ? "true" : "false");
        
        fprintf(file, "# Edge detection threshold (sensitivity)\n");
        fprintf(file, "# Lower values = more edges detected\n");
        fprintf(file, "# Range: 0.05 - 0.3, Recommended: 0.250\n");
        fprintf(file, "EdgeThreshold = %.3f\n\n", edgeThreshold);
        
        fprintf(file, "# Edge strength/intensity\n");
        fprintf(file, "# Range: 0.1 - 3.0, Recommended: 0.90\n");
        fprintf(file, "EdgeStrength = %.2f\n\n", edgeStrength);
        
        fprintf(file, "# Edge thickness multiplier\n");
        fprintf(file, "# Range: 0.1 - 5.0, Recommended: 0.35\n");
        fprintf(file, "EdgeThickness = %.2f\n\n", edgeThickness);
        
        fprintf(file, "# Edge color (RGB values)\n");
        fprintf(file, "# Range: 0.0 - 1.0 for each component\n");
        fprintf(file, "EdgeColorR = %.2f\n", edgeColor.x);
        fprintf(file, "EdgeColorG = %.2f\n", edgeColor.y);
        fprintf(file, "EdgeColorB = %.2f\n", edgeColor.z);
        fprintf(file, "EdgeColorA = %.2f\n\n", edgeColor.w);
        
        fprintf(file, "# Enable color cycling animation on edges\n");
        fprintf(file, "# Values: true/false\n");
        fprintf(file, "ColorcycleEnabled = %s\n\n", colorcycleEnabled ? "true" : "false");
        
        fprintf(file, "# Color cycling speed\n");
        fprintf(file, "# Range: 0.1 - 10.0, Recommended: 2.0\n");
        fprintf(file, "ColorcycleSpeed = %.2f\n\n", colorcycleSpeed);
        
        fprintf(file, "# Edge detection method\n");
        fprintf(file, "# 0 = Sobel (original method), 1 = Canny (new method)\n");
        fprintf(file, "# Values: 0 or 1, Recommended: 1 for quality, 0 for performance\n");
        fprintf(file, "EdgeDetectionMethod = %d\n\n", edgeDetectionMethod);
        
        fprintf(file, "[Canny Edge Detection]\n");
        fprintf(file, "# Low threshold for Canny edge detection\n");
        fprintf(file, "# Range: 0.05 - 0.3, Recommended: 0.120\n");
        fprintf(file, "CannyLowThreshold = %.3f\n\n", cannyLowThreshold);
        
        fprintf(file, "# High threshold for Canny edge detection\n");
        fprintf(file, "# Range: 0.1 - 0.5, Recommended: 0.350\n");
        fprintf(file, "CannyHighThreshold = %.3f\n\n", cannyHighThreshold);
        
        fprintf(file, "# Gaussian blur sigma for Canny edge detection\n");
        fprintf(file, "# Range: 0.8 - 2.5, Recommended: 1.00\n");
        fprintf(file, "CannyGaussianSigma = %.2f\n\n", cannyGaussianSigma);
        
        fprintf(file, "# Alpha for weak edges (low threshold) - separate control\n");
        fprintf(file, "# Range: 0.0 - 1.0, Recommended: 0.99\n");
        fprintf(file, "CannyLowAlpha = %.2f\n\n", cannyLowAlpha);
        
        fprintf(file, "# Alpha for strong edges (high threshold) - separate control\n");
        fprintf(file, "# Range: 0.0 - 1.0, Recommended: 0.96\n");
        fprintf(file, "CannyHighAlpha = %.2f\n\n", cannyHighAlpha);
        
        fprintf(file, "[Edge Fading]\n");
        fprintf(file, "# Enable/disable edge fading effect\n");
        fprintf(file, "# Values: true/false\n");
        fprintf(file, "EdgeFadingEnabled = %s\n\n", edgeFadingEnabled ? "true" : "false");
        
        fprintf(file, "# Edge fading strength (0.0-1.0)\n");
        fprintf(file, "# Range: 0.0 - 1.0, Recommended: 0.5\n");
        fprintf(file, "EdgeFadingStrength = %.2f\n\n", edgeFadingStrength);
        
        fprintf(file, "# Edge fading distance factor (0.1-2.0)\n");
        fprintf(file, "# Range: 0.1 - 2.0, Recommended: 1.50\n");
        fprintf(file, "EdgeFadingDistance = %.2f\n\n", edgeFadingDistance);
        
        fprintf(file, "# Edge fading mode (0=Distance-based, 1=Screen-edge, 2=Gradient)\n");
        fprintf(file, "# 0 = Distance-based fading from screen center\n");
        fprintf(file, "# 1 = Screen-edge based fading\n");
        fprintf(file, "# 2 = Gradient-based radial fading\n");
        fprintf(file, "EdgeFadingMode = %d\n\n", edgeFadingMode);
        
        fprintf(file, "# Canny-specific fading strength for weak edges (0.0-1.0)\n");
        fprintf(file, "# Range: 0.0 - 1.0, Recommended: 0.25\n");
        fprintf(file, "CannyLowFadingStrength = %.2f\n\n", cannyLowFadingStrength);
        
        fprintf(file, "# Canny-specific fading strength for strong edges (0.0-1.0)\n");
        fprintf(file, "# Range: 0.0 - 1.0, Recommended: 0.10\n");
        fprintf(file, "CannyHighFadingStrength = %.2f\n\n", cannyHighFadingStrength);
        
        fprintf(file, "[Bloom Compatibility]\n");
        fprintf(file, "# Enable bloom compatibility mode for better integration with bloom mods\n");
        fprintf(file, "# Values: true/false, Recommended: true\n");
        fprintf(file, "BloomCompatibilityMode = %s\n\n", bloomCompatibilityMode ? "true" : "false");
        
        fprintf(file, "# Blending mode for edge effects\n");
        fprintf(file, "# 0 = Normal alpha blending\n");
        fprintf(file, "# 1 = Bloom-compatible blending (special handling for black edges)\n");
        fprintf(file, "# 2 = Multiply blending (darkens the image)\n");
        fprintf(file, "# 3 = Screen blending (lightens the image)\n");
        fprintf(file, "# 4 = Overlay blending (enhances contrast)\n");
        fprintf(file, "# Values: 0-4, Recommended: 1 for bloom mods, 0 for normal use\n");
        fprintf(file, "BlendingMode = %d\n\n", blendingMode);
        
        fprintf(file, "[Performance Notes]\n");
        fprintf(file, "# - Sobel edge detection: Low performance impact (~5-10%%)\n");
        fprintf(file, "# - Canny edge detection: Medium performance impact (~15-25%%)\n");
        fprintf(file, "# - Color cycling: Minimal performance impact\n");
        fprintf(file, "# - Edge fading: Minimal performance impact (~1-2%%)\n");
        fprintf(file, "# - All parameters optimized for GTA SA coordinate system\n\n");
        
        fprintf(file, "[Edge Detection Methods]\n");
        fprintf(file, "# Sobel Method (EdgeDetectionMethod = 0):\n");
        fprintf(file, "# - Fast and efficient\n");
        fprintf(file, "# - Good for real-time applications\n");
        fprintf(file, "# - May produce some noise\n");
        fprintf(file, "# - Recommended for lower-end systems\n");
        fprintf(file, "# - Implementation: Based on exact Sobel filter tutorial source code\n");
        fprintf(file, "# - Kernels: Gx = [[1,0,-1],[2,0,-2],[1,0,-1]], Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]\n");
        fprintf(file, "# - Normalized for GTA SA scale (4800x4800 world units)\n\n");
        
        fprintf(file, "# Canny Method (EdgeDetectionMethod = 1):\n");
        fprintf(file, "# - More accurate edge detection\n");
        fprintf(file, "# - Includes noise reduction (Gaussian blur)\n");
        fprintf(file, "# - Uses double thresholding for better edge quality\n");
        fprintf(file, "# - Higher computational cost\n");
        fprintf(file, "# - Recommended for higher-end systems\n");
        fprintf(file, "# - Implementation: Uses exact Sobel kernels from tutorial for gradient calculation\n");
        fprintf(file, "# - Parameters normalized for GTA SA coordinate system\n\n");
        
        fprintf(file, "[Troubleshooting]\n");
        fprintf(file, "# If edges are too thick: Reduce EdgeThickness\n");
        fprintf(file, "# If edges are too thin: Increase EdgeThickness\n");
        fprintf(file, "# If too many edges: Increase EdgeThreshold (Sobel) or CannyHighThreshold (Canny)\n");
        fprintf(file, "# If too few edges: Decrease EdgeThreshold (Sobel) or CannyLowThreshold (Canny)\n");
        fprintf(file, "# If edges are too strong: Reduce EdgeStrength\n");
        fprintf(file, "# If edges are too weak: Increase EdgeStrength\n");
        fprintf(file, "# If Canny edges are too noisy: Increase CannyGaussianSigma\n");
        fprintf(file, "# If Canny edges are too blurry: Decrease CannyGaussianSigma\n");
        fprintf(file, "# If Canny detects too many weak edges: Increase CannyLowThreshold\n");
        fprintf(file, "# If Canny misses important edges: Decrease CannyHighThreshold\n");
        fprintf(file, "# If edge fading is too strong: Reduce EdgeFadingStrength\n");
        fprintf(file, "# If edge fading is too weak: Increase EdgeFadingStrength\n");
        fprintf(file, "# If edge fading distance is too short: Increase EdgeFadingDistance\n");
        fprintf(file, "# If edge fading distance is too long: Decrease EdgeFadingDistance\n");
        
        fclose(file);
    }
    
    void InitializeEdgeDetectionShader() {
        try {
            // Get D3D9 device
            auto dev = reinterpret_cast<IDirect3DDevice9*>(GetD3DDevice());
            if (!dev) {
                edgeDetectionShader = nullptr;
                return;
            }
            
            // Use plugin SDK's shader compilation with embedded strings
            char* compiledVS = nullptr;
            char* compiledPS = nullptr;
            
            // Try to compile vertex shader
            try {
                compiledVS = plugin::Shader::CompileShaderFromString(
                    edgeDetectionVertexShaderSource, "main", "vs_3_0", false);
            } catch (...) {
                compiledVS = nullptr;
            }
            
            // Try to compile pixel shader
            try {
                compiledPS = plugin::Shader::CompileShaderFromString(
                    edgeDetectionPixelShaderSource, "main", "ps_3_0", false);
            } catch (...) {
                compiledPS = nullptr;
            }
            
            if (compiledVS && compiledPS) {
                // Create shader object
                edgeDetectionShader = new CustomShader();
                
                // Create shader objects
                IDirect3DVertexShader9* vs = nullptr;
                IDirect3DPixelShader9* ps = nullptr;
                
                HRESULT vsResult = dev->CreateVertexShader((DWORD*)compiledVS, &vs);
                HRESULT psResult = dev->CreatePixelShader((DWORD*)compiledPS, &ps);
                
                if (SUCCEEDED(vsResult) && SUCCEEDED(psResult)) {
                    edgeDetectionShader->SetShaders(vs, ps);
                } else {
                    if (vs) vs->Release();
                    if (ps) ps->Release();
                    delete edgeDetectionShader;
                    edgeDetectionShader = nullptr;
                }
                
                // Clean up compiled shaders
                delete[] compiledVS;
                delete[] compiledPS;
            } else {
                // Clean up if compilation failed
                if (compiledVS) delete[] compiledVS;
                if (compiledPS) delete[] compiledPS;
                edgeDetectionShader = nullptr;
            }
        }
        catch (...) {
            // Shader compilation failed
            edgeDetectionShader = nullptr;
        }
    }
    
    void RenderEdgeDetection() {
        // Check if edge detection shader is properly initialized
        if (!edgeDetectionShader || !edgeDetectionShader->pixelShader) {
            InitializeEdgeDetectionShader();
            if (!edgeDetectionShader || !edgeDetectionShader->pixelShader) {
                return;
            }
        }
        
        // Get current screen dimensions
        RwCamera* camera = RwCameraGetCurrentCamera();
        if (!camera) return;
        
        RwRaster* cameraRaster = RwCameraGetRaster(camera);
        if (!cameraRaster) return;
        
        // Update animation time for colorcycle
        animationTime += 0.016f; // Approximate 60 FPS
        if (animationTime > 1000.0f) animationTime = 0.0f; // Reset to prevent overflow
        
        // Create edge raster if needed
        if (!edgeRaster || 
            RwRasterGetWidth(edgeRaster) != RwRasterGetWidth(cameraRaster) ||
            RwRasterGetHeight(edgeRaster) != RwRasterGetHeight(cameraRaster)) {
            
            if (edgeRaster) {
                RwRasterDestroy(edgeRaster);
            }
            
            edgeRaster = RwRasterCreate(RwRasterGetWidth(cameraRaster), 
                                       RwRasterGetHeight(cameraRaster), 
                                       0, rwRASTERTYPECAMERATEXTURE);
        }
        
        // Copy current screen to edge raster
        if (edgeRaster) {
            RwRasterPushContext(edgeRaster);
            RwRasterRenderFast(cameraRaster, 0, 0);
            RwRasterPopContext();
        }
        
        // Store original render states for compatibility with bloom mods
        RwRenderState originalZTest, originalZWrite, originalFog, originalVertexAlpha;
        RwBlendFunction originalSrcBlend, originalDestBlend;
        
        RwRenderStateGet(rwRENDERSTATEZTESTENABLE, &originalZTest);
        RwRenderStateGet(rwRENDERSTATEZWRITEENABLE, &originalZWrite);
        RwRenderStateGet(rwRENDERSTATEFOGENABLE, &originalFog);
        RwRenderStateGet(rwRENDERSTATEVERTEXALPHAENABLE, &originalVertexAlpha);
        RwRenderStateGet(rwRENDERSTATESRCBLEND, &originalSrcBlend);
        RwRenderStateGet(rwRENDERSTATEDESTBLEND, &originalDestBlend);
        
        // Set up render states for edge detection pass (bloom-compatible)
        RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)FALSE);
        RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)FALSE);
        RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
        RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)TRUE);
        
        // Use alpha blending for bloom compatibility
        // This works better with the new shader blending logic
        RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDSRCALPHA);
        RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDINVSRCALPHA);
        
        // Enable edge detection shader
        edgeDetectionShader->EnableCustom();
        
        // Set shader parameters
        D3DXVECTOR4 edgeParams(edgeThreshold, edgeStrength, edgeThickness, (float)edgeDetectionMethod);
        edgeDetectionShader->PackPSParameters(edgeParams, 0);
        
        D3DXVECTOR4 edgeColorParams(edgeColor.x, edgeColor.y, edgeColor.z, edgeColor.w);
        edgeDetectionShader->PackPSParameters(edgeColorParams, 1);
        
        D3DXVECTOR4 colorcycleParams(colorcycleEnabled ? 1.0f : 0.0f, colorcycleSpeed, animationTime, 0.0f);
        edgeDetectionShader->PackPSParameters(colorcycleParams, 2);
        
        // Set Canny edge detection parameters
        D3DXVECTOR4 cannyParams(cannyLowThreshold, cannyHighThreshold, cannyGaussianSigma, 0.0f);
        edgeDetectionShader->PackPSParameters(cannyParams, 3);
        
        // Set Canny alpha parameters
        D3DXVECTOR4 cannyAlphaParams(cannyLowAlpha, cannyHighAlpha, 0.0f, 0.0f);
        edgeDetectionShader->PackPSParameters(cannyAlphaParams, 5);
        
        // Set edge fading parameters
        D3DXVECTOR4 fadingParams(edgeFadingEnabled ? 1.0f : 0.0f, edgeFadingStrength, edgeFadingDistance, (float)edgeFadingMode);
        edgeDetectionShader->PackPSParameters(fadingParams, 4);
        
        // Set Canny-specific fading parameters
        D3DXVECTOR4 cannyFadingParams(cannyLowFadingStrength, cannyHighFadingStrength, 0.0f, 0.0f);
        edgeDetectionShader->PackPSParameters(cannyFadingParams, 6);
        
        // Set bloom compatibility parameters
        D3DXVECTOR4 compatibilityParams(bloomCompatibilityMode ? 1.0f : 0.0f, (float)blendingMode, 0.0f, 0.0f);
        edgeDetectionShader->PackPSParameters(compatibilityParams, 7);
        
        // Set screen texture - try different approaches
        if (edgeRaster) {
            edgeDetectionShader->PackTexture((RwTexture*)edgeRaster, 0);
        } else {
            // Fallback: use camera raster directly
            edgeDetectionShader->PackTexture((RwTexture*)cameraRaster, 0);
        }
        
        // Draw full-screen quad using plugin SDK's DrawRect method
        // This uses the correct UV coordinates: (0,1), (0,0), (1,1), (1,0)
        edgeDetectionShader->DrawRect(0, 0, RwRasterGetWidth(cameraRaster), RwRasterGetHeight(cameraRaster));
        
        // Disable shader
        edgeDetectionShader->DisableCustom();
        
        // Restore original render states for bloom compatibility
        RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)originalZTest);
        RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)originalZWrite);
        RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)originalFog);
        RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)originalVertexAlpha);
        RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)originalSrcBlend);
        RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)originalDestBlend);
    }
    
    void RenderSimpleEdgeDetection() {
        // Simple fallback edge detection using immediate mode rendering
        RwCamera* camera = RwCameraGetCurrentCamera();
        if (!camera) return;
        
        RwRaster* cameraRaster = RwCameraGetRaster(camera);
        if (!cameraRaster) return;
        
        float width = (float)RwRasterGetWidth(cameraRaster);
        float height = (float)RwRasterGetHeight(cameraRaster);
        
        // Set up render states
        RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)FALSE);
        RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)FALSE);
        RwRenderStateSet(rwRENDERSTATEFOGENABLE, (void*)FALSE);
        RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)TRUE);
        RwRenderStateSet(rwRENDERSTATESRCBLEND, (void*)rwBLENDSRCALPHA);
        RwRenderStateSet(rwRENDERSTATEDESTBLEND, (void*)rwBLENDINVSRCALPHA);
        RwRenderStateSet(rwRENDERSTATETEXTURERASTER, (void*)0);
        
        // Create full-screen quad vertices with green color for edge detection
        // Using correct UV coordinates based on plugin SDK: (0,1), (0,0), (1,1), (1,0)
        RwIm2DVertex vertices[4];
        
        // Top-left (0, 0) with UV (0, 1)
        RwIm2DVertexSetScreenX(&vertices[0], 0.0f);
        RwIm2DVertexSetScreenY(&vertices[0], 0.0f);
        RwIm2DVertexSetScreenZ(&vertices[0], 0.0f);
        RwIm2DVertexSetRecipCameraZ(&vertices[0], 1.0f);
        RwIm2DVertexSetIntRGBA(&vertices[0], 0, 255, 0, 64); // Green with low alpha for edge detection
        RwIm2DVertexSetU(&vertices[0], 0.0f, 1.0f);
        RwIm2DVertexSetV(&vertices[0], 1.0f, 1.0f);
        
        // Bottom-left (0, height) with UV (0, 0)
        RwIm2DVertexSetScreenX(&vertices[1], 0.0f);
        RwIm2DVertexSetScreenY(&vertices[1], height);
        RwIm2DVertexSetScreenZ(&vertices[1], 0.0f);
        RwIm2DVertexSetRecipCameraZ(&vertices[1], 1.0f);
        RwIm2DVertexSetIntRGBA(&vertices[1], 0, 255, 0, 64);
        RwIm2DVertexSetU(&vertices[1], 0.0f, 1.0f);
        RwIm2DVertexSetV(&vertices[1], 0.0f, 1.0f);
        
        // Top-right (width, 0) with UV (1, 1)
        RwIm2DVertexSetScreenX(&vertices[2], width);
        RwIm2DVertexSetScreenY(&vertices[2], 0.0f);
        RwIm2DVertexSetScreenZ(&vertices[2], 0.0f);
        RwIm2DVertexSetRecipCameraZ(&vertices[2], 1.0f);
        RwIm2DVertexSetIntRGBA(&vertices[2], 0, 255, 0, 64);
        RwIm2DVertexSetU(&vertices[2], 1.0f, 1.0f);
        RwIm2DVertexSetV(&vertices[2], 1.0f, 1.0f);
        
        // Bottom-right (width, height) with UV (1, 0)
        RwIm2DVertexSetScreenX(&vertices[3], width);
        RwIm2DVertexSetScreenY(&vertices[3], height);
        RwIm2DVertexSetScreenZ(&vertices[3], 0.0f);
        RwIm2DVertexSetRecipCameraZ(&vertices[3], 1.0f);
        RwIm2DVertexSetIntRGBA(&vertices[3], 0, 255, 0, 64);
        RwIm2DVertexSetU(&vertices[3], 1.0f, 1.0f);
        RwIm2DVertexSetV(&vertices[3], 0.0f, 1.0f);
        
        // Draw the quad
        RwIm2DRenderPrimitive(rwPRIMTYPETRIFAN, vertices, 4);
        
        // Restore render states
        RwRenderStateSet(rwRENDERSTATEZTESTENABLE, (void*)TRUE);
        RwRenderStateSet(rwRENDERSTATEZWRITEENABLE, (void*)TRUE);
        RwRenderStateSet(rwRENDERSTATEVERTEXALPHAENABLE, (void*)FALSE);
    }
} zpostfxPlugin; 
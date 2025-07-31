#include "C:/1/plugin-sdk-master/shared/PluginBase.h"
#include "C:/1/plugin-sdk-master/shared/Events.h"
#include "C:/1/plugin-sdk-master/shared/extensions/Shader.h"
#include "C:/1/plugin-sdk-master/plugin_sa/game_sa/rw/rwcore.h"
#include "C:/1/plugin-sdk-master/shared/extensions/Config.h"
#include "C:\1\plugin-sdk-master\shared\RenderWare.h"
#include "C:/1/plugin-sdk-master/plugin_sa/game_sa/common.h"
#include <algorithm>
#include <cmath>
#include <windows.h>
#include "C:/1/plugin-sdk-master/DXSDK/Include/d3dx9.h"
#include "C:/1/plugin-sdk-master/DXSDK/Include/d3dx9math.h"
#include "ShaderCode.h"

using std::min;
using std::max;

using namespace plugin;

// Custom shader class for embedded shader strings
class CustomShader {
public:
    IDirect3DPixelShader9* pixelShader = nullptr;
    IDirect3DVertexShader9* vertexShader = nullptr;
    
    CustomShader() {
        pixelShader = nullptr;
        vertexShader = nullptr;
    }
    
    ~CustomShader() {
        if (vertexShader) {
            vertexShader->Release();
            vertexShader = nullptr;
        }
        if (pixelShader) {
            pixelShader->Release();
            pixelShader = nullptr;
        }
    }
    
    void Enable() {
        if (vertexShader) _rwD3D9SetVertexShader(vertexShader);
        if (pixelShader) _rwD3D9SetPixelShader(pixelShader);
    }
    
    void Disable() {
        _rwD3D9SetVertexShader(0);
        _rwD3D9SetPixelShader(0);
    }
    
    // Use Plugin SDK's parameter packing methods
    template <class T> bool PackPSParameters(T& parameters, unsigned int offset = 0) {
        auto dev = reinterpret_cast<IDirect3DDevice9*>(GetD3DDevice());
        unsigned int fsize = sizeof(parameters);
        if (fsize % 16) {
            unsigned int newSize = fsize + 16 - (fsize % 16);
            char *newData = new char[newSize];
            memset(newData, 0, newSize);
            memcpy(newData, &parameters, fsize);
            dev->SetPixelShaderConstantF(offset, (float *)newData, newSize / 16);
            delete[] newData;
        }
        else
            dev->SetPixelShaderConstantF(offset, (float *)&parameters, fsize / 16);
        return true;
    }
    
    // Texture packing methods
    void PackTexture(RwTexture* texture, unsigned int idx) {
        RwD3D9SetTexture(texture, idx);
    }
    
    void PackTexture(RpMaterial* material, unsigned int idx) {
        if (material && material->texture)
            RwD3D9SetTexture(material->texture, idx);
        else
            RwD3D9SetTexture(0, idx);
    }
    
    void PackTexture(RxD3D9InstanceData* mesh, unsigned int idx) {
        if (mesh && mesh->material && mesh->material->texture)
            RwD3D9SetTexture(mesh->material->texture, idx);
        else
            RwD3D9SetTexture(0, idx);
    }
    
    // Draw full-screen quad method
    void DrawRect(float left, float top, float right, float bottom) {
        D3DVIEWPORT9 oldViewport, viewport;
        auto dev = reinterpret_cast<IDirect3DDevice9*>(GetD3DDevice());
        dev->GetViewport(&oldViewport);
        viewport.X = static_cast<DWORD>(left);
        viewport.Y = static_cast<DWORD>(top);
        viewport.Width = static_cast<DWORD>(right - left);
        viewport.Height = static_cast<DWORD>(bottom - top);
        viewport.MinZ = 0.0f;
        viewport.MaxZ = 1.0f;
        dev->SetViewport(&viewport);
        IDirect3DVertexDeclaration9*  VertDecl = NULL, *oldVertDecl = NULL;
        struct Vertex {
            D3DXVECTOR2 pos;
            D3DXVECTOR2 tex_coord;
        } quad[4];
        quad[0].pos = D3DXVECTOR2(-1, -1); quad[0].tex_coord = D3DXVECTOR2(0, 1);
        quad[1].pos = D3DXVECTOR2(-1, 1);  quad[1].tex_coord = D3DXVECTOR2(0, 0);
        quad[2].pos = D3DXVECTOR2(1, -1);  quad[2].tex_coord = D3DXVECTOR2(1, 1);
        quad[3].pos = D3DXVECTOR2(1, 1);   quad[3].tex_coord = D3DXVECTOR2(1, 0);
        const D3DVERTEXELEMENT9 Decl[] = {
            { 0, 0,  D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
            { 0, 8, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
            D3DDECL_END()
        };
        dev->CreateVertexDeclaration(Decl, &VertDecl);
        dev->GetVertexDeclaration(&oldVertDecl);
        dev->SetVertexDeclaration(VertDecl);
        _rwD3D9DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(Vertex));
        VertDecl->Release();
        dev->SetVertexDeclaration(oldVertDecl);
        dev->SetViewport(&oldViewport);
    }
};

class zpostfx {
private:
    // Configuration file
    config_file config;
    

    
    // Edge Detection Settings - Optimized for GTA SA
    // Edge effect is always enabled (no configuration needed)
    float edgeStrength = 0.65f; // Edge intensity (0.1-3.0) - Slightly transparent for better quality
    
    // Canny Edge Detection Parameters (optimized for GTA SA)
    float cannyLowThreshold = 0.2f; // Low threshold (0.05-0.3) - More sensitive for better edge detection
    float cannyHighThreshold = 0.3f; // High threshold (0.1-0.5) - Balanced for quality
    float cannyGaussianSigma = 1.5f; // Blur amount (0.8-2.5) - Smoother edges
    
    // Edge Color Settings - Configurable

    

    
    // Blending Settings (automatic bloom detection)
    bool bloomCompatibilityMode = false; // Enable bloom compatibility (automatic detection)
    
    // Distance-Based Threshold Adjustment (automatic optimization) - Highest Quality
    bool dynamicThresholdEnabled = true;
    float distanceThresholdMultiplier = 1.25f; // Distance sensitivity (0.5-3.0) - Enhanced for better depth perception
    float minDistanceThreshold = 0.15f; // Close objects threshold (0.05-1.0) - More sensitive for close details
    float maxDistanceThreshold = 0.08f; // Distant objects threshold (0.05-1.0) - Better distant edge detection
    
    // Depth Buffer Settings (automatic optimization)
    bool useDepthBuffer = true; // Enable depth buffer access (automatic fallback if unavailable)
    
    // Alpha controls for edge transparency
    float lowContrastAlpha = 0.8f; // Alpha for low contrast edges (0.1-1.0)
    float highContrastAlpha = 0.9f; // Alpha for high contrast edges (0.1-1.0)
    
    // Automatic parameters (not configurable) - Highest Quality
    float cannyLowAlpha = 0.998f; // Automatic alpha for weak edges - Higher quality
    float cannyHighAlpha = 0.99f; // Automatic alpha for strong edges - Higher quality
    int blendingMode = 5; // Blending mode (0-5, optimized for bloom/non-bloom)
    float distanceSensitivity = 1.5f; // Automatic distance sensitivity - Enhanced for better depth perception
    float baseDistance = 1200.0f; // Automatic base distance - Optimized for GTA SA world scale
    float nearPlane = 0.015f; // Automatic near plane - Better near-field detail
    float farPlane = 1500.0f; // Automatic far plane - Extended range for large maps
    // Contrast-based thickness is always enabled (no configuration needed)
    float lowContrastThickness = 0.2f; // Thickness for low contrast areas (0.1-5.0) - Enhanced visibility
    float highContrastThickness = 0.25f; // Thickness for high contrast areas (0.1-5.0) - Sharper edges
    float contrastThreshold = 0.15f; // Threshold to determine low vs high contrast (0.05-0.5) - More sensitive

    
    // Shader object
    CustomShader* edgeDetectionShader = nullptr;
    
    // Raster objects for rendering
    RwRaster* edgeRaster = nullptr;
    
    // Time tracking for animations (kept for potential future use)
    float animationTime = 0.0f;

        // Shader source code references (defined in ShaderCode.h)
    const char* edgeDetectionVertexShaderSource = EdgeFXShaders::edgeDetectionVertexShaderSource;
    
    // Combined pixel shader source code
    std::string GetPixelShaderSource() {
        std::string source = EdgeFXShaders::edgeDetectionPixelShaderSource_Part1;
        source += EdgeFXShaders::edgeDetectionPixelShaderSource_Part2;
        source += EdgeFXShaders::edgeDetectionPixelShaderSource_Part3;
        source += EdgeFXShaders::edgeDetectionPixelShaderSource_Part4;
        return source;
    }

public:
    zpostfx() {
        // Suppress shader compilation error messages by redirecting stderr temporarily
        freopen("nul", "w", stderr);
        
        // Load configuration
        LoadConfig();
        
        // Initialize shaders
        InitializeEdgeDetectionShader();
        
        // Restore stderr
        freopen("CON", "w", stderr);
        
        // Hook into rendering events
        Events::drawingEvent.after += [this]() {
            // Render edge detection effect (always enabled)
                RenderEdgeDetection();
        };
    }
    
    ~zpostfx() {
        if (edgeDetectionShader) {
            delete edgeDetectionShader;
        }

        if (edgeRaster) {
            RwRasterDestroy(edgeRaster);
        }
    }
    

    
    void LoadConfig() {
        config.open("EdgeFX.ini");
        
        // Edge effect is always enabled (no configuration needed)
        edgeStrength = config["EdgeStrength"].asFloat(0.65f);
        
        // Contrast-based thickness is always enabled (no configuration needed)
        lowContrastThickness = config["LowContrastThickness"].asFloat(0.2f);
        highContrastThickness = config["HighContrastThickness"].asFloat(0.25f);
        contrastThreshold = config["ContrastThreshold"].asFloat(0.15f);
        
        // Load Canny edge detection configuration
        cannyLowThreshold = config["CannyLowThreshold"].asFloat(0.2f);
        cannyHighThreshold = config["CannyHighThreshold"].asFloat(0.3f);
        cannyGaussianSigma = config["CannyGaussianSigma"].asFloat(1.5f);
        
        // Load alpha controls for edge transparency
        lowContrastAlpha = config["LowContrastAlpha"].asFloat(0.8f);
        highContrastAlpha = config["HighContrastAlpha"].asFloat(0.9f);
        

        
        // Load bloom compatibility settings
        bloomCompatibilityMode = config["BloomCompatibilityMode"].asBool(false);
        
        // Load blending mode (0-5, optimized for bloom/non-bloom)
        blendingMode = config["BlendingMode"].asInt(5);
        // Ensure blending mode is within valid range (0-5)
        blendingMode = max(0, min(5, blendingMode));
        
        // Load dynamic distance-based threshold settings (multiplier is now automatic)
        dynamicThresholdEnabled = config["DynamicThresholdEnabled"].asBool(true);
        minDistanceThreshold = config["MinDistanceThreshold"].asFloat(0.15f);
        maxDistanceThreshold = config["MaxDistanceThreshold"].asFloat(0.08f);
        // Ensure values are within valid range (0.05-1.0)
        minDistanceThreshold = max(0.05f, min(1.0f, minDistanceThreshold));
        maxDistanceThreshold = max(0.05f, min(1.0f, maxDistanceThreshold));
        
        // Load depth buffer settings (always enabled by default)
        useDepthBuffer = config["UseDepthBuffer"].asBool(true);
        

        
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
        
        // If file doesn't exist, create it with the new configuration
        FILE* file = fopen("EdgeFX.ini", "w");
        if (!file) return;
        
        fprintf(file, "# EdgeFX Configuration File\n");
        fprintf(file, "# This file contains settings for edge detection effects\n\n");
        
        fprintf(file, "# Edge strength/intensity\n");
        fprintf(file, "# Range: 0.1 - 3.0, Recommended: 1.0\n");
        fprintf(file, "# Note: Values below 1.0 create transparent black edges with alpha equal to strength\n");
        fprintf(file, "EdgeStrength = 0.65\n\n");
        
        fprintf(file, "# Low threshold for Canny edge detection\n");
        fprintf(file, "# Range: 0.05 - 0.3, Recommended: 0.08\n");
        fprintf(file, "CannyLowThreshold = 0.2\n\n");
        
        fprintf(file, "# High threshold for Canny edge detection\n");
        fprintf(file, "# Range: 0.1 - 0.5, Recommended: 0.18 \n");
        fprintf(file, "CannyHighThreshold = 0.3\n\n");
        
        fprintf(file, "# Thickness for low contrast areas\n");
        fprintf(file, "# Range: 0.1 - 5.0, Recommended: 0.4\n");
        fprintf(file, "LowContrastThickness = 0.2\n\n");
        
        fprintf(file, "# Thickness for high contrast areas\n");
        fprintf(file, "# Range: 0.1 - 5.0, Recommended: 0.25\n");
        fprintf(file, "HighContrastThickness = 0.25\n\n");
        
        fprintf(file, "# Alpha/transparency for low contrast edges\n");
        fprintf(file, "# Range: 0.1 - 1.0, Recommended: 0.8\n");
        fprintf(file, "# Lower values = more transparent edges\n");
        fprintf(file, "LowContrastAlpha = 0.8\n\n");
        
        fprintf(file, "# Alpha/transparency for high contrast edges\n");
        fprintf(file, "# Range: 0.1 - 1.0, Recommended: 0.9\n");
        fprintf(file, "# Lower values = more transparent edges\n");
        fprintf(file, "HighContrastAlpha = 0.9\n\n");
        
        fprintf(file, "# Threshold to determine low vs high contrast\n");
        fprintf(file, "# Range: 0.05 - 0.5, Recommended: 0.12\n");
        fprintf(file, "ContrastThreshold = 0.15\n\n");
        
        fprintf(file, "# Gaussian blur sigma for Canny edge detection\n");
        fprintf(file, "# Range: 0.8 - 2.5, Recommended: 1.1\n");
        fprintf(file, "CannyGaussianSigma = 1.5\n\n");
        
        fprintf(file, "[Blending Modes]\n");
        fprintf(file, "# Advanced blending mode for edge rendering (optimized for bloom/non-bloom)\n");
        fprintf(file, "# Values: 0-5, Recommended: 0 (pure black edges)\n");
        fprintf(file, "# 0: Normal alpha blending (pure black edges, best for non-bloom)\n");
        fprintf(file, "# 1: Bloom-compatible mode (darker edges, best for bloom mods)\n");
        fprintf(file, "# 2: Multiply blend (intense dark edges, good for both)\n");
        fprintf(file, "# 3: Screen blend (lighter edges, good for dark scenes)\n");
        fprintf(file, "# 4: Overlay blend (enhanced contrast, good for both)\n");
        fprintf(file, "# 5: Soft light blend (subtle enhancement, good for both)\n");
        fprintf(file, "BlendingMode = 5\n\n");
        
        fprintf(file, "[Bloom Compatibility]\n");
        fprintf(file, "# Enable bloom compatibility mode for better integration with bloom mods\n");
        fprintf(file, "# Values: true/false, Recommended: false (pure black edges)\n");
        fprintf(file, "BloomCompatibilityMode = false\n\n");
        
        fprintf(file, "[Dynamic Threshold]\n");
        fprintf(file, "# Enable dynamic threshold adjustment based on screen distance\n");
        fprintf(file, "# Values: true/false\n");
        fprintf(file, "DynamicThresholdEnabled = true\n\n");
        
        fprintf(file, "# Minimum threshold value for close objects (screen center)\n");
        fprintf(file, "# Range: 0.05 - 1.0, Recommended: 0.15\n");
        fprintf(file, "# Lower values = more edges detected for close objects\n");
        fprintf(file, "MinDistanceThreshold = 0.15\n\n");
        
        fprintf(file, "# Maximum threshold value for distant objects (screen edges)\n");
        fprintf(file, "# Range: 0.05 - 1.0, Recommended: 0.08\n");
        fprintf(file, "# Higher values = fewer edges detected for distant objects\n");
        fprintf(file, "MaxDistanceThreshold = 0.08\n\n");
        
        fprintf(file, "[Advanced Distance Calculation]\n");
        fprintf(file, "# Enable depth buffer access for more accurate distance calculation\n");
        fprintf(file, "# Values: true/false, Recommended: true\n");
        fprintf(file, "UseDepthBuffer = true\n");
        
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
            
            // Create shader object
            edgeDetectionShader = new CustomShader();
            
            // Compile vertex shader from embedded string
            char* compiledVS = plugin::Shader::CompileShaderFromString(
                    edgeDetectionVertexShaderSource, "main", "vs_3_0", false);
            
            // Compile pixel shader from embedded string
            std::string pixelShaderSource = GetPixelShaderSource();
            char* compiledPS = plugin::Shader::CompileShaderFromString(
                pixelShaderSource.c_str(), "main", "ps_3_0", false);
            
            // Check if compilation failed
            if (!compiledVS || !compiledPS) {
                if (compiledVS) delete[] compiledVS;
                if (compiledPS) delete[] compiledPS;
                delete edgeDetectionShader;
                edgeDetectionShader = nullptr;
                return;
            }
                
                // Create shader objects
            HRESULT vsResult = dev->CreateVertexShader((DWORD*)compiledVS, &edgeDetectionShader->vertexShader);
            HRESULT psResult = dev->CreatePixelShader((DWORD*)compiledPS, &edgeDetectionShader->pixelShader);
                
                // Clean up compiled shaders
                delete[] compiledVS;
                delete[] compiledPS;
            
            // Check if creation failed
            if (FAILED(vsResult) || FAILED(psResult)) {
                delete edgeDetectionShader;
                edgeDetectionShader = nullptr;
                return;
            }
            
            // Set texture filtering to nearest neighbor for pixelated look
            dev->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            dev->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            dev->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
        }
        catch (...) {
            // Shader compilation failed
            if (edgeDetectionShader) {
                delete edgeDetectionShader;
            edgeDetectionShader = nullptr;
            }
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
        
        // Update animation time (kept for potential future use)
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
        edgeDetectionShader->Enable();
        
        // Set texture filtering to nearest neighbor (point filtering) for pixelated look
        auto dev = reinterpret_cast<IDirect3DDevice9*>(GetD3DDevice());
        if (dev) {
            dev->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            dev->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            dev->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);
        }
        
        // Set shader parameters with proper error checking
        D3DXVECTOR4 edgeParams(0.0f, edgeStrength, 0.0f, 0.0f); // x = unused, y = strength, z = unused, w = unused
        if (!edgeDetectionShader->PackPSParameters(edgeParams, 0)) {
            return; // Failed to set parameters
        }
        
        // Set Canny edge detection parameters
        D3DXVECTOR4 cannyParams(cannyLowThreshold, cannyHighThreshold, cannyGaussianSigma, 0.0f);
        if (!edgeDetectionShader->PackPSParameters(cannyParams, 3)) {
            return; // Failed to set parameters
        }
        
        // Set Canny alpha parameters
        D3DXVECTOR4 cannyAlphaParams(lowContrastAlpha, highContrastAlpha, 0.0f, 0.0f);
        if (!edgeDetectionShader->PackPSParameters(cannyAlphaParams, 5)) {
            return; // Failed to set parameters
        }
        
        
        
        // Set bloom compatibility parameters
        D3DXVECTOR4 compatibilityParams(bloomCompatibilityMode ? 1.0f : 0.0f, (float)blendingMode, 0.0f, 0.0f);
        if (!edgeDetectionShader->PackPSParameters(compatibilityParams, 7)) {
            return; // Failed to set parameters
        }
        
        // Set dynamic distance-based threshold parameters
        D3DXVECTOR4 distanceParams(dynamicThresholdEnabled ? 1.0f : 0.0f, distanceThresholdMultiplier, distanceSensitivity, baseDistance);
        if (!edgeDetectionShader->PackPSParameters(distanceParams, 8)) {
            return; // Failed to set parameters
        }
        
        // Set threshold range parameters
        D3DXVECTOR4 thresholdRange(minDistanceThreshold, maxDistanceThreshold, 0.0f, 0.0f);
        if (!edgeDetectionShader->PackPSParameters(thresholdRange, 9)) {
            return; // Failed to set parameters
        }
        
        // Set advanced distance calculation parameters
        // Distance calculation is now automatic - no manual parameters needed
        
        // Set depth buffer parameters
        D3DXVECTOR4 depthParams(nearPlane, farPlane, useDepthBuffer ? 1.0f : 0.0f, 0.0f);
        if (!edgeDetectionShader->PackPSParameters(depthParams, 11)) {
            return; // Failed to set parameters
        }
        
        // Set contrast-based thickness parameters (always enabled)
        D3DXVECTOR4 contrastThicknessParams(1.0f, lowContrastThickness, highContrastThickness, contrastThreshold);
        if (!edgeDetectionShader->PackPSParameters(contrastThicknessParams, 12)) {
            return; // Failed to set parameters
        }
        
        // Set screen texture with proper error handling
        if (edgeRaster) {
            edgeDetectionShader->PackTexture((RwTexture*)edgeRaster, 0);
        } else {
            // Fallback: use camera raster directly
            edgeDetectionShader->PackTexture((RwTexture*)cameraRaster, 0);
        }
        
        // Depth buffer handling - use screen texture as fallback for safety
        // This ensures compatibility and prevents crashes
        edgeDetectionShader->PackTexture((RwTexture*)cameraRaster, 1);
        
        // Disable depth buffer mode for safety (use screen-based distance calculation)
        D3DXVECTOR4 depthCoverageParams(0.0f, 0.0f, 0.0f, 0.0f);
        if (!edgeDetectionShader->PackPSParameters(depthCoverageParams, 13)) {
            return; // Failed to set parameters
        }
        
        // Draw full-screen quad using plugin SDK's DrawRect method
        // This uses the correct UV coordinates: (0,1), (0,0), (1,1), (1,0)
        edgeDetectionShader->DrawRect(0, 0, RwRasterGetWidth(cameraRaster), RwRasterGetHeight(cameraRaster));
        
        // Disable shader
        edgeDetectionShader->Disable();
        
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
};

// Plugin instance
zpostfx zpostfxPlugin; 
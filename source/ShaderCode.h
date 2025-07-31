#pragma once

// Edge Detection Shader Code
// This file contains the shader source code to avoid "string too big" compilation errors

namespace EdgeFXShaders {
    
    // Vertex shader source code
    const char* const edgeDetectionVertexShaderSource = R"(
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

    // Pixel shader source code - Part 1 (Uniforms and Structures)
    const char* const edgeDetectionPixelShaderSource_Part1 = R"(
uniform sampler2D screenTexture : register(s0);
uniform sampler2D depthTexture : register(s1);
uniform float4 edgeParams : register(c0); // x = unused, y = strength, z = unused, w = unused
uniform float4 cannyParams : register(c3); // x = lowThreshold, y = highThreshold, z = gaussianSigma, w = unused
uniform float4 cannyAlphaParams : register(c5); // x = lowAlpha, y = highAlpha, z = unused, w = unused
uniform float4 compatibilityParams : register(c7); // x = bloomCompatibilityMode, y = blendingMode, z = unused, w = unused
uniform float4 distanceParams : register(c8); // x = enabled, y = multiplier, z = sensitivity, w = baseDistance
uniform float4 thresholdRange : register(c9); // x = minThreshold, y = maxThreshold, z = unused, w = unused
// Distance calculation is now automatic - no manual parameters needed
uniform float4 depthParams : register(c11); // x = nearPlane, y = farPlane, z = useDepthBuffer, w = unused
uniform float4 contrastThicknessParams : register(c12); // x = enabled, y = lowThickness, z = highThickness, w = contrastThreshold
uniform float4 depthCoverageParams : register(c13); // x = enabled, y = unused, z = unused, w = unused

struct PS_INPUT {
    float2 texcoord : TEXCOORD0;
};

// Linearize depth value from depth buffer
float linearizeDepth(float depth, float nearPlane, float farPlane) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

// Calculate local contrast for contrast-based thickness control
float calculateLocalContrast(sampler2D tex, float2 texcoord, float2 texelSize) {
    float4 topLeft = tex2D(tex, texcoord + float2(-texelSize.x, -texelSize.y));
    float4 top = tex2D(tex, texcoord + float2(0.0, -texelSize.y));
    float4 topRight = tex2D(tex, texcoord + float2(texelSize.x, -texelSize.y));
    float4 left = tex2D(tex, texcoord + float2(-texelSize.x, 0.0));
    float4 center = tex2D(tex, texcoord);
    float4 right = tex2D(tex, texcoord + float2(texelSize.x, 0.0));
    float4 bottomLeft = tex2D(tex, texcoord + float2(-texelSize.x, texelSize.y));
    float4 bottom = tex2D(tex, texcoord + float2(0.0, texelSize.y));
    float4 bottomRight = tex2D(tex, texcoord + float2(texelSize.x, texelSize.y));
    
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
    
    float mean = (tl + t + tr + l + c + r + bl + b + br) / 9.0;
    float variance = 0.0;
    variance += (tl - mean) * (tl - mean);
    variance += (t - mean) * (t - mean);
    variance += (tr - mean) * (tr - mean);
    variance += (l - mean) * (l - mean);
    variance += (c - mean) * (c - mean);
    variance += (r - mean) * (r - mean);
    variance += (bl - mean) * (bl - mean);
    variance += (b - mean) * (b - mean);
    variance += (br - mean) * (br - mean);
    
    return sqrt(variance / 9.0);
}
)";

    // Pixel shader source code - Part 2 (Distance calculation functions)
    const char* const edgeDetectionPixelShaderSource_Part2 = R"(
// Advanced distance calculation with full screen coverage using automatic hybrid approach
float calculateAdvancedDistance(float2 texcoord, float4 distanceParams, float4 depthParams) {
    float2 center = float2(0.5, 0.5);
    float2 offset = texcoord - center;
    
    // Automatic screen-space distance calculations (always provide full coverage)
    // These represent distance from screen center, not camera distance
    float perspectiveDistance = length(offset) * (1.0 + length(offset) * 1.2); // Automatic perspective correction
    float radialDistance = length(offset);
    float exponentialDistance = 1.0 - exp(-radialDistance * 2.8);
    float manhattanDistance = abs(offset.x) + abs(offset.y);
    float chebyshevDistance = max(abs(offset.x), abs(offset.y));
    
    // Automatic blending of screen-space distance methods
    float screenDistance = lerp(perspectiveDistance, exponentialDistance, 0.7); // Automatic distance blending
    screenDistance = lerp(screenDistance, manhattanDistance, 0.6); // Automatic depth buffer weight
    screenDistance = lerp(screenDistance, chebyshevDistance, 0.4); // Automatic screen distance weight
    
    // Advanced depth buffer handling with full screen coverage
    float depthDistance = screenDistance; // Default to screen distance for full coverage
    float depthWeight = 0.0; // Weight for depth buffer influence
    
    if (depthParams.z > 0.5) { // useDepthBuffer enabled
        // Advanced depth buffer handling with full screen coverage
        if (depthCoverageParams.x > 0.5) { // depth coverage mode enabled
            // Use sophisticated depth coverage approach for full screen coverage
            float4 depthCoverageSample = tex2D(depthTexture, texcoord);
            
            // Convert depth coverage data to distance with full screen coverage
            float depthCoverageValue = depthCoverageSample.r;
            
            // Apply sophisticated depth-to-distance conversion with coverage
            float normalizedDepthCoverage = saturate(depthCoverageValue);
            depthDistance = 1.0 - normalizedDepthCoverage;
            
            // Calculate advanced depth weight with full coverage consideration
            float depthCoverageQuality = 0.0;
            
            // Sample neighboring depth coverage values for quality assessment
            float4 leftCoverage = tex2D(depthTexture, texcoord + float2(-0.001, 0.0));
            float4 rightCoverage = tex2D(depthTexture, texcoord + float2(0.001, 0.0));
            float4 topCoverage = tex2D(depthTexture, texcoord + float2(0.0, -0.001));
            float4 bottomCoverage = tex2D(depthTexture, texcoord + float2(0.0, 0.001));
            
            // Calculate depth coverage gradient for quality assessment
            float coverageGradient = abs(depthCoverageValue - leftCoverage.r) + 
                                   abs(depthCoverageValue - rightCoverage.r) + 
                                   abs(depthCoverageValue - topCoverage.r) + 
                                   abs(depthCoverageValue - bottomCoverage.r);
            
            // Normalize coverage gradient
            coverageGradient = saturate(coverageGradient * 2.5);
            
            // Calculate depth coverage quality based on gradient and screen position
            float coverageConsistency = 1.0 - coverageGradient; // Higher consistency = better quality
            float screenPositionQuality = 1.0 - saturate(length(offset) * 1.3); // Better quality near center
            
            // Combine quality factors for depth coverage
            depthCoverageQuality = coverageConsistency * screenPositionQuality;
            
            // Calculate adaptive depth weight with automatic coverage blending
            float baseDepthWeight = 0.65; // Automatic depth buffer weight
            float adaptiveCoverageWeight = depthCoverageQuality * baseDepthWeight;
            
            // Apply edge-aware weight reduction to prevent circular artifacts
            float edgeReduction = 1.0 - saturate(length(offset) * 1.8);
            depthWeight = adaptiveCoverageWeight * edgeReduction;
            
            // Ensure minimum screen distance influence for full coverage
            depthWeight = min(depthWeight, 0.75); // Cap depth influence
            
            // Apply sophisticated depth coverage distance enhancement
            // Use depth coverage data to enhance screen distance calculation
            float enhancedCoverageDistance = lerp(depthDistance, screenDistance, 0.25);
            depthDistance = enhancedCoverageDistance;
            
        } else {
            // Fallback to screen texture-based depth approximation
            float4 screenDepthSample = tex2D(depthTexture, texcoord);
            
            // Convert screen color to depth-like value using luminance
            float3 weights = float3(0.2126, 0.7152, 0.0722);
            float screenDepthValue = dot(screenDepthSample.rgb, weights);
            
            // Apply screen-based depth distance calculation
            float screenLuminanceDepth = 1.0 - screenDepthValue; // Invert so brighter = closer
            
            // Calculate screen-based depth weight
            float screenDepthQuality = 0.0;
            
            // Sample neighboring screen pixels for quality assessment
            float4 leftScreen = tex2D(depthTexture, texcoord + float2(-0.001, 0.0));
            float4 rightScreen = tex2D(depthTexture, texcoord + float2(0.001, 0.0));
            float4 topScreen = tex2D(depthTexture, texcoord + float2(0.0, -0.001));
            float4 bottomScreen = tex2D(depthTexture, texcoord + float2(0.0, 0.001));
            
            float leftLum = dot(leftScreen.rgb, weights);
            float rightLum = dot(rightScreen.rgb, weights);
            float topLum = dot(topScreen.rgb, weights);
            float bottomLum = dot(bottomScreen.rgb, weights);
            
            // Calculate screen content variation for quality assessment
            float screenVariation = abs(screenDepthValue - leftLum) + 
                                  abs(screenDepthValue - rightLum) + 
                                  abs(screenDepthValue - topLum) + 
                                  abs(screenDepthValue - bottomLum);
            
            screenVariation = saturate(screenVariation * 2.0);
            
            // Calculate screen-based depth quality
            float screenConsistency = 1.0 - screenVariation;
            float screenPositionQuality = 1.0 - saturate(length(offset) * 1.4);
            
            screenDepthQuality = screenConsistency * screenPositionQuality;
            
            // Calculate screen-based depth weight
            float baseScreenWeight = 0.65 * 0.6; // Automatic reduced weight for screen-based approach
            float adaptiveScreenWeight = screenDepthQuality * baseScreenWeight;
            
            // Apply edge-aware weight reduction
            float screenEdgeReduction = 1.0 - saturate(length(offset) * 1.6);
            depthWeight = adaptiveScreenWeight * screenEdgeReduction;
            
            // Ensure minimum screen distance influence
            depthWeight = min(depthWeight, 0.6); // Lower cap for screen-based approach
            
            // Apply screen-based depth distance enhancement
            float enhancedScreenDistance = lerp(screenLuminanceDepth, screenDistance, 0.3);
            depthDistance = enhancedScreenDistance;
        }
    }
    
    // Final distance calculation with adaptive blending
    float finalDistance = lerp(screenDistance, depthDistance, depthWeight);
    
    // Ensure full screen coverage by clamping to reasonable range
    finalDistance = saturate(finalDistance);
    
    return finalDistance;
}

// Advanced dynamic threshold calculation with correct distance-based adjustment
float calculateDynamicThreshold(float2 texcoord, float baseThreshold, float4 distanceParams, float4 thresholdRange) {
    if (distanceParams.x < 0.5) return baseThreshold;
    
    float multiplier = distanceParams.y;
    float sensitivity = distanceParams.z;
    float baseDist = distanceParams.w;
    float minThreshold = thresholdRange.x;
    float maxThreshold = thresholdRange.y;
    
    // Calculate distance from camera (closer = lower distance value)
    float advancedDistance = calculateAdvancedDistance(texcoord, distanceParams, depthParams);
    
    // Normalize distance with sensitivity (closer objects = lower normalized distance)
    float normalizedDistance = saturate(advancedDistance * sensitivity);
    
    // Apply non-linear transformation for better distance perception
    float transformedDistance = pow(normalizedDistance, 1.4);
    
    // Calculate distance factor (closer objects = lower factor, distant objects = higher factor)
    float distanceFactor = smoothstep(0.0, 1.0, transformedDistance);
    
    // Apply multiplier to distance factor
    float adjustedDistanceFactor = saturate(distanceFactor * multiplier);
    
    // FIXED: The issue was that screen distance was being used instead of actual camera distance
    // For proper distance-based threshold adjustment:
    // - Close objects (low distance) should have lower thresholds (more edges)
    // - Distant objects (high distance) should have higher thresholds (fewer edges)
    
    // Use screen distance as a fallback when depth buffer is not available
    float2 center = float2(0.5, 0.5);
    float2 offset = texcoord - center;
    float screenDistance = length(offset);
    
    // Blend between depth-based distance and screen distance
    float finalDistance = lerp(advancedDistance, screenDistance, 0.3);
    
    // Normalize the final distance
    float normalizedFinalDistance = saturate(finalDistance * sensitivity);
    float transformedFinalDistance = pow(normalizedFinalDistance, 1.4);
    float finalDistanceFactor = smoothstep(0.0, 1.0, transformedFinalDistance);
    
    // Apply multiplier to final distance factor
    float adjustedFinalDistanceFactor = saturate(finalDistanceFactor * multiplier);
    
    // Handle both normal and inverted threshold ranges (min can be higher than max)
    float actualMin = min(minThreshold, maxThreshold);
    float actualMax = max(minThreshold, maxThreshold);
    bool isInverted = minThreshold > maxThreshold;
    
    // Calculate the lerp factor based on whether the range is inverted
    float lerpFactor = isInverted ? (1.0 - adjustedFinalDistanceFactor) : adjustedFinalDistanceFactor;
    
    // CORRECTED: Closer objects should use minThreshold, distant objects should use maxThreshold
    float adjustedThreshold = lerp(actualMin, actualMax, lerpFactor);
    
    // Calculate blend factor for smooth transition
    float blendFactor = lerp(0.65, 0.85, finalDistanceFactor);
    
    // Blend between base threshold and adjusted threshold
    float finalThreshold = lerp(baseThreshold, adjustedThreshold, blendFactor);
    
    // Ensure threshold stays within valid range (using actual min/max)
    return clamp(finalThreshold, actualMin, actualMax);
}
)";

    // Pixel shader source code - Part 3 (Gaussian blur and Canny edge detection)
    const char* const edgeDetectionPixelShaderSource_Part3 = R"(
// Optimized Gaussian blur function for Canny edge detection
float4 gaussianBlur(sampler2D tex, float2 texcoord, float2 texelSize, float sigma) {
    float4 result = float4(0.0, 0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    
    float weights[5] = {0.06136, 0.24477, 0.38774, 0.24477, 0.06136};
    
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            float2 offset = float2(i, j) * texelSize;
            float weight = weights[i+2] * weights[j+2];
            result += tex2D(tex, texcoord + offset) * weight;
            totalWeight += weight;
        }
    }
    
    return result / totalWeight;
}

// Canny edge detection using correct Sobel implementation with separate alpha control
float3 cannyEdgeDetection(sampler2D tex, float2 texcoord, float2 texelSize, float lowThreshold, float highThreshold, float sigma, float lowAlpha, float highAlpha) {
    float4 blurred = gaussianBlur(tex, texcoord, texelSize, sigma);
    
    float4 topLeft = tex2D(tex, texcoord + float2(-texelSize.x, -texelSize.y));
    float4 top = tex2D(tex, texcoord + float2(0.0, -texelSize.y));
    float4 topRight = tex2D(tex, texcoord + float2(texelSize.x, -texelSize.y));
    float4 left = tex2D(tex, texcoord + float2(-texelSize.x, 0.0));
    float4 center = tex2D(tex, texcoord);
    float4 right = tex2D(tex, texcoord + float2(texelSize.x, 0.0));
    float4 bottomLeft = tex2D(tex, texcoord + float2(-texelSize.x, texelSize.y));
    float4 bottom = tex2D(tex, texcoord + float2(0.0, texelSize.y));
    float4 bottomRight = tex2D(tex, texcoord + float2(texelSize.x, texelSize.y));
    
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
    
    float gx = 1.0*tl + 0.0*t + (-1.0)*tr + 
               2.0*l + 0.0*c + (-2.0)*r + 
               1.0*bl + 0.0*b + (-1.0)*br;
    
    float gy = 1.0*tl + 2.0*t + 1.0*tr + 
               0.0*l + 0.0*c + 0.0*r + 
               (-1.0)*bl + (-2.0)*b + (-1.0)*br;
    
    float magnitude = sqrt(gx*gx + gy*gy);
    
    float edgeValue = 0.0;
    float edgeAlpha = 0.0;
    float edgeType = 0.0;
    
    if (magnitude > highThreshold) {
        edgeValue = 1.0;
        edgeAlpha = highAlpha;
        edgeType = 2.0;
    } else if (magnitude > lowThreshold) {
        edgeValue = 1.0;
        edgeAlpha = lowAlpha;
        edgeType = 1.0;
    }
    
    return float3(edgeValue, edgeAlpha, edgeType);
}
)";

    // Pixel shader source code - Part 4 (Main function)
    const char* const edgeDetectionPixelShaderSource_Part4 = R"(
float4 main(PS_INPUT input) : COLOR {
    float4 screenColor = tex2D(screenTexture, input.texcoord);
    
    float2 texelSize = float2(0.0010, 0.0010); // Higher resolution for better quality
    
    // Use contrast-based thickness (always enabled)
    float localContrast = calculateLocalContrast(screenTexture, input.texcoord, texelSize);
    float contrastFactor = saturate(localContrast / contrastThicknessParams.w);
    float thickness = lerp(contrastThicknessParams.y, contrastThicknessParams.z, contrastFactor);
    float2 adjustedTexelSize = texelSize * thickness;
    
    float edgeValue = 0.0;
    float edgeAlpha = 0.975f;
    float edgeType = 0.0;
    
    float dynamicThreshold = calculateDynamicThreshold(input.texcoord, cannyParams.x, distanceParams, thresholdRange);
    
    float thresholdScale = dynamicThreshold / cannyParams.x;
    float dynamicLowThreshold = cannyParams.x * thresholdScale;
    float dynamicHighThreshold = cannyParams.y * thresholdScale;
    
    float3 cannyResult = cannyEdgeDetection(screenTexture, input.texcoord, adjustedTexelSize, 
                                           dynamicLowThreshold, dynamicHighThreshold, cannyParams.z,
                                           cannyAlphaParams.x, cannyAlphaParams.y);
    edgeValue = cannyResult.x * edgeParams.y;
    edgeAlpha = cannyResult.y;
    
    float4 finalColor = screenColor;
    if (edgeValue > 0.0) {
        // Use the proper alpha value from edge detection, not the edge value
        float edgeAlphaValue = edgeAlpha;
        
        // Apply edge strength to the alpha for proper intensity control
        edgeAlphaValue = edgeAlphaValue * edgeParams.y;
        
        float4 currentEdgeColor = float4(0.0, 0.0, 0.0, edgeAlphaValue);
        float blendAlpha = edgeAlphaValue;
        int blendMode = int(compatibilityParams.y);
        
        // Ensure blend mode is within valid range (0-5)
        blendMode = clamp(blendMode, 0, 5);
        
        if (blendMode == 0) {
            // Normal alpha blending (pure black edges, best for non-bloom)
            finalColor.rgb = lerp(finalColor.rgb, float3(0.0, 0.0, 0.0), blendAlpha);
        }
        else if (blendMode == 1) {
            // Bloom-compatible mode (darker edges, best for bloom mods) - Enhanced quality
            finalColor.rgb = finalColor.rgb * (1.0 - blendAlpha * 0.9);
        }
        else if (blendMode == 2) {
            // Multiply blend (intense dark edges, good for both) - Enhanced quality
            finalColor.rgb = lerp(finalColor.rgb, finalColor.rgb * float3(0.0, 0.0, 0.0), blendAlpha * 1.1);
        }
        else if (blendMode == 3) {
            // Screen blend (lighter edges, good for dark scenes) - Enhanced quality
            float3 edgeIntensity = float3(blendAlpha * 0.5, blendAlpha * 0.5, blendAlpha * 0.5);
            float3 screenBlend = 1.0 - (1.0 - edgeIntensity) * (1.0 - finalColor.rgb);
            finalColor.rgb = lerp(finalColor.rgb, screenBlend, blendAlpha);
        }
        else if (blendMode == 4) {
            // Overlay blend (enhanced contrast, good for both)
            float3 overlayBlend;
            overlayBlend.r = (finalColor.r < 0.5) ? (2.0 * finalColor.r * 0.0) : (1.0 - 2.0 * (1.0 - finalColor.r) * (1.0 - 0.0));
            overlayBlend.g = (finalColor.g < 0.5) ? (2.0 * finalColor.g * 0.0) : (1.0 - 2.0 * (1.0 - finalColor.g) * (1.0 - 0.0));
            overlayBlend.b = (finalColor.b < 0.5) ? (2.0 * finalColor.b * 0.0) : (1.0 - 2.0 * (1.0 - finalColor.b) * (1.0 - 0.0));
            finalColor.rgb = lerp(finalColor.rgb, overlayBlend, blendAlpha);
        }
        else if (blendMode == 5) {
            // Soft light blend (subtle enhancement, good for both)
            float3 softLightBlend;
            softLightBlend.r = (0.0 < 0.5) ? (2.0 * 0.0 * finalColor.r) : (1.0 - 2.0 * (1.0 - 0.0) * (1.0 - finalColor.r));
            softLightBlend.g = (0.0 < 0.5) ? (2.0 * 0.0 * finalColor.g) : (1.0 - 2.0 * (1.0 - 0.0) * (1.0 - finalColor.g));
            softLightBlend.b = (0.0 < 0.5) ? (2.0 * 0.0 * finalColor.b) : (1.0 - 2.0 * (1.0 - 0.0) * (1.0 - finalColor.b));
            finalColor.rgb = lerp(finalColor.rgb, softLightBlend, blendAlpha);
        }
        
        finalColor.rgb = saturate(finalColor.rgb);
    }
    
    return finalColor;
}
)";

} // namespace EdgeFXShaders 
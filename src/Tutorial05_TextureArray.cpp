/*
 *  Copyright 2019-2024 Diligent Graphics LLC
 *  Copyright 2015-2019 Egor Yusov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  In no event and under no legal theory, whether in tort (including negligence),
 *  contract, or otherwise, unless required by applicable law (such as deliberate
 *  and grossly negligent acts) or agreed to in writing, shall any Contributor be
 *  liable for any damages, including any direct, indirect, special, incidental,
 *  or consequential damages of any character arising as a result of this License or
 *  out of the use or inability to use the software (including but not limited to damages
 *  for loss of goodwill, work stoppage, computer failure or malfunction, or any and
 *  all other commercial damages or losses), even if such Contributor has been advised
 *  of the possibility of such damages.
 */

#include <random>
#include <string>

#include "Tutorial05_TextureArray.hpp"
#include "MapHelper.hpp"
#include "GraphicsUtilities.h"
#include "TextureUtilities.h"
#include "ColorConversion.h"
#include "../../Common/src/TexturedCube.hpp"
#include "imgui.h"

namespace Diligent
{

SampleBase* CreateSample()
{
    return new Tutorial05_TextureArray();
}

namespace
{

struct InstanceData
{
    float4x4 Matrix;
    float    TextureInd = 0;
};

} // namespace

void Tutorial05_TextureArray::CreatePipelineState()
{
    // clang-format off
    // Define vertex shader input layout
    // This tutorial uses two types of input: per-vertex data and per-instance data.
    LayoutElement LayoutElems[] =
    {
        // Per-vertex data - first buffer slot
        // Attribute 0 - vertex position
        LayoutElement{0, 0, 3, VT_FLOAT32, False},
        // Attribute 1 - texture coordinates
        LayoutElement{1, 0, 2, VT_FLOAT32, False},

        // Per-instance data - second buffer slot
        // We will use four attributes to encode instance-specific 4x4 transformation matrix
        // Attribute 2 - first row
        LayoutElement{2, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
        // Attribute 3 - second row
        LayoutElement{3, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
        // Attribute 4 - third row
        LayoutElement{4, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
        // Attribute 5 - fourth row
        LayoutElement{5, 1, 4, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
        // Attribute 6 - texture array index
        LayoutElement{6, 1, 1, VT_FLOAT32, False, INPUT_ELEMENT_FREQUENCY_PER_INSTANCE},
    };
    // clang-format on

    // Create a shader source stream factory to load shaders from files.
    RefCntAutoPtr<IShaderSourceInputStreamFactory> pShaderSourceFactory;
    m_pEngineFactory->CreateDefaultShaderSourceStreamFactory(nullptr, &pShaderSourceFactory);

    TexturedCube::CreatePSOInfo CubePsoCI;
    CubePsoCI.pDevice                = m_pDevice;
    CubePsoCI.RTVFormat              = m_pSwapChain->GetDesc().ColorBufferFormat;
    CubePsoCI.DSVFormat              = m_pSwapChain->GetDesc().DepthBufferFormat;
    CubePsoCI.pShaderSourceFactory   = pShaderSourceFactory;
    CubePsoCI.VSFilePath             = "cube_inst.vsh";
    CubePsoCI.PSFilePath             = "cube_inst.psh";
    CubePsoCI.ExtraLayoutElements    = LayoutElems;
    CubePsoCI.NumExtraLayoutElements = _countof(LayoutElems);

    m_pPSO = TexturedCube::CreatePipelineState(CubePsoCI, m_ConvertPSOutputToGamma);

    // Create dynamic uniform buffer that will store our transformation matrix
    // Dynamic buffers can be frequently updated by the CPU
    CreateUniformBuffer(m_pDevice, sizeof(float4x4) * 2, "VS constants CB", &m_VSConstants);

    // Since we did not explicitly specify the type for 'Constants' variable, default
    // type (SHADER_RESOURCE_VARIABLE_TYPE_STATIC) will be used. Static variables
    // never change and are bound directly to the pipeline state object.
    m_pPSO->GetStaticVariableByName(SHADER_TYPE_VERTEX, "Constants")->Set(m_VSConstants);

    // Since we are using mutable variable, we must create a shader resource binding object
    // http://diligentgraphics.com/2016/03/23/resource-binding-model-in-diligent-engine-2-0/
    m_pPSO->CreateShaderResourceBinding(&m_SRB, true);
}

void Tutorial05_TextureArray::CreateInstanceBuffer()
{
    // Create instance data buffer that will store transformation matrices
    BufferDesc InstBuffDesc;
    InstBuffDesc.Name = "Instance data buffer";
    // Use default usage as this buffer will only be updated when grid size changes
    InstBuffDesc.Usage     = USAGE_DEFAULT;
    InstBuffDesc.BindFlags = BIND_VERTEX_BUFFER;
    InstBuffDesc.Size      = sizeof(InstanceData) * MaxInstances;
    m_pDevice->CreateBuffer(InstBuffDesc, nullptr, &m_InstanceBuffer);
    PopulateInstanceBuffer();
}

void Tutorial05_TextureArray::LoadTextures()
{
    std::vector<RefCntAutoPtr<ITextureLoader>> TexLoaders(NumTextures);
    // Load textures
    for (int tex = 0; tex < NumTextures; ++tex)
    {
        // Create loader for the current texture
        std::stringstream FileNameSS;
        FileNameSS << "DGLogo" << tex << ".png";
        const auto      FileName = FileNameSS.str();
        TextureLoadInfo LoadInfo;
        LoadInfo.IsSRGB = true;

        CreateTextureLoaderFromFile(FileName.c_str(), IMAGE_FILE_FORMAT_UNKNOWN, LoadInfo, &TexLoaders[tex]);
        VERIFY_EXPR(TexLoaders[tex]);
        VERIFY(tex == 0 || TexLoaders[tex]->GetTextureDesc() == TexLoaders[0]->GetTextureDesc(), "All textures must be same size");
    }

    auto TexArrDesc      = TexLoaders[0]->GetTextureDesc();
    TexArrDesc.ArraySize = NumTextures;
    TexArrDesc.Type      = RESOURCE_DIM_TEX_2D_ARRAY;
    TexArrDesc.Usage     = USAGE_DEFAULT;
    TexArrDesc.BindFlags = BIND_SHADER_RESOURCE;

    // Prepare initialization data
    std::vector<TextureSubResData> SubresData(TexArrDesc.ArraySize * TexArrDesc.MipLevels);
    for (Uint32 slice = 0; slice < TexArrDesc.ArraySize; ++slice)
    {
        for (Uint32 mip = 0; mip < TexArrDesc.MipLevels; ++mip)
        {
            SubresData[slice * TexArrDesc.MipLevels + mip] = TexLoaders[slice]->GetSubresourceData(mip, 0);
        }
    }
    TextureData InitData{SubresData.data(), TexArrDesc.MipLevels * TexArrDesc.ArraySize};

    // Create the texture array
    RefCntAutoPtr<ITexture> pTexArray;
    m_pDevice->CreateTexture(TexArrDesc, &InitData, &pTexArray);

    // Get shader resource view from the texture array
    m_TextureSRV = pTexArray->GetDefaultView(TEXTURE_VIEW_SHADER_RESOURCE);
    // Set texture SRV in the SRB
    m_SRB->GetVariableByName(SHADER_TYPE_PIXEL, "g_Texture")->Set(m_TextureSRV);
}

void Tutorial05_TextureArray::UpdateUI()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        if (ImGui::SliderInt("Grid Size", &m_GridSize, 1, 32))
        {
            PopulateInstanceBuffer();
        }
    }
    ImGui::End();
}

void Tutorial05_TextureArray::Initialize(const SampleInitInfo& InitInfo)
{
    SampleBase::Initialize(InitInfo);

    CreatePipelineState();

    // Load cube vertex and index buffers
    m_CubeVertexBuffer = TexturedCube::CreateVertexBuffer(m_pDevice, GEOMETRY_PRIMITIVE_VERTEX_FLAG_POS_TEX);
    m_CubeIndexBuffer  = TexturedCube::CreateIndexBuffer(m_pDevice);

    CreateInstanceBuffer();
    LoadTextures();
}

static float angle = (PI_F / 1.0);

void Tutorial05_TextureArray::PopulateInstanceBuffer()
{
    {
        const Uint32              NumInstances = 22;
        std::vector<InstanceData> InstanceDataArray(NumInstances);

        angle += 0.003f;


        InstanceDataArray[0].Matrix = float4x4::Scale(5.0f, 0.1f, 0.01f) * float4x4::Translation(0.0f, 0.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[0].TextureInd = 3;

        InstanceDataArray[1].Matrix = float4x4::Scale(0.01f, 0.1f, 5.0f) * float4x4::Translation(0.0f, 0.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[1].TextureInd = 3;

        InstanceDataArray[2].Matrix = float4x4::Scale(0.1f, 1.0f, 0.01f) * float4x4::Translation(-5.0f, -1.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[2].TextureInd = 3;

        InstanceDataArray[3].Matrix = float4x4::Scale(0.1f, 1.0f, 0.01f) * float4x4::Translation(5.0f, -1.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[3].TextureInd = 3;

        InstanceDataArray[4].Matrix = float4x4::Scale(0.1f, 1.0f, 0.01f) * float4x4::Translation(0.0f, 1.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[4].TextureInd = 3;

        InstanceDataArray[5].Matrix = float4x4::Scale(0.05f, 1.0f, 0.01f) * float4x4::Translation(0.0f, -1.0f, -5.0f) * float4x4::RotationY(angle);
        InstanceDataArray[5].TextureInd = 3;

        InstanceDataArray[6].Matrix = float4x4::Scale(0.05f, 1.0f, 0.01f) * float4x4::Translation(0.0f, -1.0f, 5.0f) * float4x4::RotationY(angle);
        InstanceDataArray[6].TextureInd = 3;

        InstanceDataArray[7].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(-5.0f, -2.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[7].TextureInd = 0;

        InstanceDataArray[8].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(5.0f, -2.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[8].TextureInd = 2;

        InstanceDataArray[9].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(0.0f, -2.0f, -5.0f) * float4x4::RotationY(angle);
        InstanceDataArray[9].TextureInd = 1;

        InstanceDataArray[10].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(0.0f, -2.0f, 5.0f) * float4x4::RotationY(angle);
        InstanceDataArray[10].TextureInd = 0;

        InstanceDataArray[11].Matrix = float4x4::Scale(3.0f, 0.05f, 0.01f) * float4x4::Translation(0.0f, -5.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[11].TextureInd = 3;

        InstanceDataArray[12].Matrix = float4x4::Scale(0.01f, 0.05f, 3.0f) * float4x4::Translation(0.0f, -5.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[12].TextureInd = 3;

        InstanceDataArray[13].Matrix = float4x4::Scale(0.05f, 4.0f, 0.01f) * float4x4::Translation(0.0f, -1.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[13].TextureInd = 3;

        InstanceDataArray[14].Matrix = float4x4::Scale(0.05f, 1.0f, 0.01f) * float4x4::Translation(-3.0f, -6.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[14].TextureInd = 3;

        InstanceDataArray[15].Matrix = float4x4::Scale(0.05f, 1.0f, 0.01f) * float4x4::Translation(3.0f, -6.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[15].TextureInd = 3;

        InstanceDataArray[16].Matrix = float4x4::Scale(0.05f, 1.0f, 0.01f) * float4x4::Translation(0.0f, -6.0f, 3.0f) * float4x4::RotationY(angle);
        InstanceDataArray[16].TextureInd = 3;

        InstanceDataArray[17].Matrix = float4x4::Scale(0.05f, 1.0f, 0.01f) * float4x4::Translation(0.0f, -6.0f, -3.0f) * float4x4::RotationY(angle);
        InstanceDataArray[17].TextureInd = 3;

        InstanceDataArray[18].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(-3.0f, -7.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[18].TextureInd = 1;

        InstanceDataArray[19].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(3.0f, -7.0f, 0.0f) * float4x4::RotationY(angle);
        InstanceDataArray[19].TextureInd = 0;

        InstanceDataArray[20].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(0.0f, -7.0f, 3.0f) * float4x4::RotationY(angle);
        InstanceDataArray[20].TextureInd = 2;

        InstanceDataArray[21].Matrix = float4x4::Scale(1, 1, 1) * float4x4::Translation(0.0f, -7.0f, -3.0f) * float4x4::RotationY(angle);
        InstanceDataArray[21].TextureInd = 1;

        Uint32 DataSize = static_cast<Uint32>(sizeof(InstanceData) * InstanceDataArray.size());
        m_pImmediateContext->UpdateBuffer(m_InstanceBuffer, 0, DataSize, InstanceDataArray.data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    }
}


// Render a frame
void Tutorial05_TextureArray::Render()
{
    auto* pRTV = m_pSwapChain->GetCurrentBackBufferRTV();
    auto* pDSV = m_pSwapChain->GetDepthBufferDSV();


    // Clear the back buffer
    float4 ClearColor = {0.0f, 0.0f, 0.0f, 1.0f};

    PopulateInstanceBuffer();
    if (m_ConvertPSOutputToGamma)
    {
        // If manual gamma correction is required, we need to clear the render target with sRGB color
        ClearColor = LinearToSRGB(ClearColor);
    }
    m_pImmediateContext->ClearRenderTarget(pRTV, ClearColor.Data(), RESOURCE_STATE_TRANSITION_MODE_TRANSITION);
    m_pImmediateContext->ClearDepthStencil(pDSV, CLEAR_DEPTH_FLAG, 1.f, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    {
        // Map the buffer and write current world-view-projection matrix
        MapHelper<float4x4> CBConstants(m_pImmediateContext, m_VSConstants, MAP_WRITE, MAP_FLAG_DISCARD);
        CBConstants[0] = m_ViewProjMatrix;
        CBConstants[1] = m_RotationMatrix;
    }

    // Bind vertex, instance and index buffers
    const Uint64 offsets[] = {0, 0};
    IBuffer*     pBuffs[]  = {m_CubeVertexBuffer, m_InstanceBuffer};
    m_pImmediateContext->SetVertexBuffers(0, _countof(pBuffs), pBuffs, offsets, RESOURCE_STATE_TRANSITION_MODE_TRANSITION, SET_VERTEX_BUFFERS_FLAG_RESET);
    m_pImmediateContext->SetIndexBuffer(m_CubeIndexBuffer, 0, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    // Set the pipeline state
    m_pImmediateContext->SetPipelineState(m_pPSO);
    // Commit shader resources. RESOURCE_STATE_TRANSITION_MODE_TRANSITION mode
    // makes sure that resources are transitioned to required states.
    m_pImmediateContext->CommitShaderResources(m_SRB, RESOURCE_STATE_TRANSITION_MODE_TRANSITION);

    DrawIndexedAttribs DrawAttrs;       // This is an indexed draw call
    DrawAttrs.IndexType    = VT_UINT32; // Index type
    DrawAttrs.NumIndices   = 36;
    DrawAttrs.NumInstances = m_GridSize * m_GridSize * m_GridSize; // The number of instances
    // Verify the state of vertex and index buffers
    DrawAttrs.Flags = DRAW_FLAG_VERIFY_ALL;
    m_pImmediateContext->DrawIndexed(DrawAttrs);
}

void Tutorial05_TextureArray::Update(double CurrTime, double ElapsedTime)
{
    SampleBase::Update(CurrTime, ElapsedTime);

    static float  yaw      = 0.0f;
    static float  pitch    = 0.0f;
    static float  distance = 20.0f;
    static float3 target(0.0f, -4.0f, 0.0f);

    const float sensitivity = 0.005f;

    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
    {
        ImVec2 dragDelta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        yaw += dragDelta.x * sensitivity;
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
    }

    const float pitchLimit = 1.57f * 0.99f;
    if (pitch > pitchLimit)
        pitch = pitchLimit;
    if (pitch < -pitchLimit)
        pitch = -pitchLimit;

    float wheel = ImGui::GetIO().MouseWheel;
    if (fabs(wheel) > 0.0f)
    {
        distance -= wheel * 2.0f;
        if (distance < 1.0f) distance = 1.0f;
        if (distance > 100.0f) distance = 100.0f;
    }

    float3 offset;
    offset.x = distance * cosf(pitch) * sinf(yaw);
    offset.y = distance * sinf(pitch);
    offset.z = distance * cosf(pitch) * cosf(yaw);

    float3 cameraPos = target + offset;
    float3 forward   = normalize(target - cameraPos);
    float3 right     = normalize(cross(float3(0.0f, 1.0f, 0.0f), forward));
    float3 camUp     = cross(forward, right);

    float panSpeed = 5.0f * static_cast<float>(ElapsedTime);
    if (ImGui::IsKeyDown(ImGuiKey_UpArrow))
        target += camUp * panSpeed;
    if (ImGui::IsKeyDown(ImGuiKey_DownArrow))
        target -= camUp * panSpeed;
    if (ImGui::IsKeyDown(ImGuiKey_RightArrow))
        target += right * panSpeed;
    if (ImGui::IsKeyDown(ImGuiKey_LeftArrow))
        target -= right * panSpeed;

    cameraPos = target + offset;

    float4x4 View;
    View._11 = right.x;
    View._12 = camUp.x;
    View._13 = forward.x;
    View._14 = 0.0f;
    View._21 = right.y;
    View._22 = camUp.y;
    View._23 = forward.y;
    View._24 = 0.0f;
    View._31 = right.z;
    View._32 = camUp.z;
    View._33 = forward.z;
    View._34 = 0.0f;
    View._41 = -dot(right, cameraPos);
    View._42 = -dot(camUp, cameraPos);
    View._43 = -dot(forward, cameraPos);
    View._44 = 1.0f;

    auto SrfPreTransform = GetSurfacePretransformMatrix(float3{0, 0, 1});
    auto Proj            = GetAdjustedProjectionMatrix(PI_F / 4.0f, 0.1f, 100.f);

    m_ViewProjMatrix = View * SrfPreTransform * Proj;
    m_RotationMatrix = float4x4::Identity();

    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 140, 10), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(600, 200), ImGuiCond_Always);
    ImGui::Begin("View Controls", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::Text("View Orientation");

    ImGui::Separator();

    // Fila 1: Diagonales Superiores
    ImGui::Text("Top Diagonal Views");
    if (ImGui::Button("Front-Right"))
    {
        yaw   = PI_F / 4.0f;
        pitch = PI_F / 4.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Top-Right"))
    {
        yaw   = 0.0f;
        pitch = PI_F / 4.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Front-Left"))
    {
        yaw   = -PI_F / 4.0f;
        pitch = PI_F / 4.0f;
    }

    ImGui::Separator();

    // Fila 2: Vistas Principales
    ImGui::Text("Main Views");
    if (ImGui::Button("Right"))
    {
        yaw   = PI_F / 2.0f;
        pitch = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Up"))
    {
        yaw   = 0.0f;
        pitch = PI_F / 2.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Front"))
    {
        yaw   = 0.0f;
        pitch = 0.0f;
    }

    if (ImGui::Button("Left"))
    {
        yaw   = -PI_F / 2.0f;
        pitch = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Down"))
    {
        yaw   = 0.0f;
        pitch = -PI_F / 2.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Back"))
    {
        yaw   = PI_F;
        pitch = 0.0f;
    }

    ImGui::Separator();

    // Fila 3: Diagonales Inferiores
    ImGui::Text("Bottom Diagonal Views");
    if (ImGui::Button("Right-Bottom"))
    {
        yaw   = PI_F / 4.0f;
        pitch = -PI_F / 4.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Down-Left"))
    {
        yaw   = 0.0f;
        pitch = -PI_F / 4.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Left-Bottom"))
    {
        yaw   = -PI_F / 4.0f;
        pitch = -PI_F / 4.0f;
    }

    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiCond_Always);
    ImGui::Begin("Controles", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    ImGui::Text("Controles de la camara:");
    ImGui::Text("- Click derecho: Rotar");
    ImGui::Text("- Flechas: Desplazarse");
    ImGui::Text("- Rueda del mouse: Zoom");
    ImGui::End();
}

} // namespace Diligent

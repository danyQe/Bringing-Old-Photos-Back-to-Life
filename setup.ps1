# Create necessary directories
New-Item -ItemType Directory -Force -Path "Face_Enhancement/models/networks"
New-Item -ItemType Directory -Force -Path "Global/detection_models"
New-Item -ItemType Directory -Force -Path "Face_Detection"
New-Item -ItemType Directory -Force -Path "Global/models"

Write-Host "Downloading Synchronized BatchNorm..."
# Download Synchronized BatchNorm
Set-Location "Face_Enhancement/models/networks"
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
Copy-Item -Recurse -Force "Synchronized-BatchNorm-PyTorch/sync_batchnorm" .
Set-Location "../../../"

Set-Location "Global/detection_models"
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
Copy-Item -Recurse -Force "Synchronized-BatchNorm-PyTorch/sync_batchnorm" .
Set-Location "../../"

Write-Host "Downloading face landmark detection model..."
# Download face landmark detection model
Set-Location "Face_Detection"
Invoke-WebRequest -Uri "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -OutFile "shape_predictor_68_face_landmarks.dat.bz2"
# Decompress bz2 file
& "C:\Program Files\7-Zip\7z.exe" x "shape_predictor_68_face_landmarks.dat.bz2" -y
Remove-Item "shape_predictor_68_face_landmarks.dat.bz2"
Set-Location "../"

Write-Host "Downloading face enhancement models..."
# Download face enhancement models
Set-Location "Face_Enhancement"
Invoke-WebRequest -Uri "https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Face_Enhancement/checkpoints.zip" -OutFile "checkpoints.zip"
Expand-Archive -Path "checkpoints.zip" -DestinationPath "." -Force
Remove-Item "checkpoints.zip"
Set-Location "../"

Write-Host "Downloading global models..."
# Download global models
Set-Location "Global"
Invoke-WebRequest -Uri "https://facevc.blob.core.windows.net/zhanbo/old_photo/pretrain/Global/checkpoints.zip" -OutFile "checkpoints.zip"
Expand-Archive -Path "checkpoints.zip" -DestinationPath "." -Force
Remove-Item "checkpoints.zip"
Set-Location "../"

Write-Host "Downloading colorization models..."
# Download colorization models
Set-Location "Global/models"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt" -OutFile "colorization_deploy_v2.prototxt"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_release_v2.caffemodel" -OutFile "colorization_release_v2.caffemodel"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy" -OutFile "pts_in_hull.npy"
Set-Location "../../"

Write-Host "Setup completed successfully!" 
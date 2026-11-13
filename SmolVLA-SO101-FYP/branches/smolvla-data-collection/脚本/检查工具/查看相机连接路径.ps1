$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$presentDevices = @(Get-PnpDevice -PresentOnly)
$deviceLookup = @{}
foreach ($device in $presentDevices) { $deviceLookup[$device.InstanceId] = $device }
$cameraResults = @()
foreach ($camera in $presentDevices | Where-Object { $_.Class -eq 'Camera' }) {
    $chain = @()
    $deviceId = $camera.InstanceId
    for ($depth = 0; $depth -lt 12 -and $deviceId; $depth++) {
        $device = $deviceLookup[$deviceId]
        $chain += [PSCustomObject]@{ name = $device.FriendlyName; instance_id = $deviceId }
        if ($deviceId -like 'PCI\*') { break }
        $parent = Get-PnpDeviceProperty -InstanceId $deviceId -KeyName 'DEVPKEY_Device_Parent' -ErrorAction SilentlyContinue
        $deviceId = $parent.Data
    }
    $locations = Get-PnpDeviceProperty -InstanceId $camera.InstanceId -KeyName 'DEVPKEY_Device_LocationPaths' -ErrorAction SilentlyContinue
    $cameraResults += [PSCustomObject]@{
        name = $camera.FriendlyName
        instance_id = $camera.InstanceId
        location_paths = $locations.Data
        controller_id = ($chain | Where-Object { $_.instance_id -like 'PCI\*' } | Select-Object -First 1).instance_id
        parent_chain = $chain
    }
}
$external = @($cameraResults | Where-Object { $_.name -in @('1080P USB Camera', 'icspring camera') })
$controllers = @($external.controller_id | Where-Object { $_ } | Select-Object -Unique)
$report = [PSCustomObject]@{
    checked_at = (Get-Date -Format o)
    external_camera_count = $external.Count
    external_controller_count = $controllers.Count
    external_cameras_share_controller = ($external.Count -eq 2 -and $controllers.Count -eq 1)
    cameras = $cameraResults
}
$json = $report | ConvertTo-Json -Depth 8
$json | Set-Content -LiteralPath (Join-Path $projectRoot '运行记录/相机连接拓扑.json') -Encoding utf8
$json

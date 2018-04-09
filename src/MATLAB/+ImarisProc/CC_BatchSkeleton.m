%Skeleton for Imaris batch in Matlab

%ImarisLib.jar needs to be in the same folder.
%Imaris needs to be running when executing the following code 

%CC 2018-01-03 tested with Imaris 9.1.0 

%Get the image folder. Only read *.ims images.
infolder = uigetdir;
files = [infolder '/*.ims'];
listing = dir(files);
nfiles = size(listing,1);

%open files in Imaris sequentially
for i = 1:nfiles
    
    filename = [infolder '/' listing(i).name];
    filename = sprintf(filename);
    vImarisApplication = StartImaris;
    vImarisApplication.FileOpen(filename,'');
    
    %get dataset in Matlab
    vDataSet = vImarisApplication.GetDataSet;
    
    %apply median filter to dataset
    vImarisApplication.GetImageProcessing.MedianFilterChannel(vDataSet,0,[5 5 5]);
    
    %create Surfaces
    ip = vImarisApplication.GetImageProcessing;
    vNewSurfaces = ip.DetectSurfaces(vDataSet, [], 0, 1, 0, true, 0, '');
    vNewSurfaces.SetName(sprintf('New Surface'));
    vImarisApplication.GetSurpassScene.AddChild(vNewSurfaces,-1);
    
    %get Surface stats 
    vSurpassComponent = vImarisApplication.GetSurpassSelection;
    vImarisObject = vImarisApplication.GetFactory.ToSurfaces(vSurpassComponent);
    vAllStatistics = vImarisObject.GetStatistics;
    vNames = cell(vAllStatistics.mNames);
    vValues = vAllStatistics.mValues;
    disp(unique(vNames))
    
    %save ims file
    newFilename = strcat(filename(1:end-4),'new.ims');
    vImarisApplication.FileSave(newFilename,'');
    
    pause(5);
    
    %Clear java handles to clear up memory and prevent future errors
    clear 'Imaris/IApplicationPrxHelper';
    clear 'Imaris/IDataSetPrxHelper';
    clear 'Imaris/IDataContainerPrxHelper';
    clear 'Imaris/IDataItemPrxHelper';
    clear 'Imaris/cStatisticValues';
    clear 'ImarisLib';
    clear 'vSurpassScene';
    clear 'vDataSet';
    clear 'vAllStatistics';
    clear 'err';
end

%Quit Imaris Application after all is done
vImarisApplication.SetVisible(~vImarisApplication.GetVisible);
vImarisApplication.Quit;

function aImarisApplication = StartImaris
    javaaddpath ImarisLib.jar;
    vImarisLib = ImarisLib;
    server = vImarisLib.GetServer();
    id = server.GetObjectID(0);
    aImarisApplication = vImarisLib.GetApplication(id);
    disp(id)
end
root_dir = [pwd, '/tbdt/baseline/'];
sub_dirs = dir(root_dir);
total = length(sub_dirs);

for i = 3:total 
    filepath = fullfile(root_dir,sub_dirs(i).name,[sub_dirs(i).name,'_001.txt']);
    targetpath1 = fullfile(root_dir,sub_dirs(i).name,[sub_dirs(i).name,'_002.txt']);
    targetpath2 = fullfile(root_dir,sub_dirs(i).name,[sub_dirs(i).name,'_003.txt']);
    copyfile(filepath , targetpath1);
    copyfile(filepath , targetpath2);
end
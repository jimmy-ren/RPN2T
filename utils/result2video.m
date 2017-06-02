addpath output/

codecList = {'Archival', 'Motion JPEG AVI', 'Motion JPEG 2000', 'MPEG-4', 'Uncompressed AVI'};
codec = codecList{2};
writerObj = VideoWriter('output.avi', codec);
writerObj.FrameRate = 24;
open(writerObj);

base_dir = 'output/save_bolt1/';
listing = dir(strcat(base_dir, '*.png'));

for m = 1:length(listing)
    I = imread(strcat(base_dir, num2str(m), '.png'));
    writeVideo(writerObj, I);

    fprintf('Writing frame %d/%d\n', m-1, length(listing));
    
end

close(writerObj);

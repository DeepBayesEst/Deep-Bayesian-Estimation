% Extract coordinate points by trajectory

% Extract 3D coordinate data from LT6 files, storing them on a per-trajectory basis

disp('Starting to extract coordinate values!!!')

for i = 1:4

    % Convert char data to string
    temp_i = num2str(i);
    
    % Root directory (relative path)
    root_str = 'wh_data/';
    
    % Dynamically update the folder name
    folder_name = 'SFO2006Q1-';
    
    % Complete the updated folder name
    temp_str = strcat(folder_name,temp_i);
    
    % Concatenate strings to form the relative path for storing files
    relative_path = strcat(root_str,temp_str);
    
    % Iterate through the files in the folder
    file = dir(relative_path);
    len = length(file);
    disp(len)
    % The loop condition is set because there are different files at the start and end of the folder
    for j = 3:len-2
        
        % In this loop, open each file and use the extract_track_txt function to extract 3D coordinate information
        
        % Get the name of each LT6 file
        file_txt_name = file(j).name;
        
        % Split the string
        temp = split(file_txt_name,'.');
        
        % The relative path of each LT6 file
        a = strcat('/',file_txt_name);
        path = strcat(relative_path,a);
        
        % Call the function extract_track_txt_new to extract 3D coordinate points from the txt file
        [lines,row,Total] = extract_track_txt_new(path);
        
        % The stored name is ¡®*.mat¡¯
        % store_name = strcat(temp{1,1},'.mat');
        
        save(['matÊý¾Ý\',temp{1,1}],'Total');
        disp('Extraction successful!!!')
    end
    
end

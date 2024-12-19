function [lines,row,Total] = extract_track_txt_new(path)
 
% Extract three-dimensional space coordinate points from lt6 files
 
% path£ºExtracted file path
 
% every_lt6£ºIt stores three-dimensional space coordinate points
 
      % Open the file (the following paths can be used to open it)
      % f = fopen('.\wh_data\SFO2006Q1-1\20060101.lt6');
      % f = fopen('wh_data\SFO2006Q1-1\20060101.lt6');
      
      % Opening a file
      file_id = fopen(path);
 
      % The mat matrix that stores the three-dimensional coordinate points of each lt6 file
      every_lt6 = {};
      
      % Initialize the number of lines in the file
      lines = 0;
      
      % Stores the number of rows of the matrix every_lt6
      row = 1;
      linestart = 0;
      % Flight number
      n_track = 0;
       strArray = string.empty;
      % Control variable
      flag = 1;
      % feof£ºCheck if it is the end of the file
      while ~feof(file_id)
          
          % Read each line
          file = fgetl(file_id);
          
          % When a string contains the TRACK character, before the second occurrence of the TRACK character, this is an independent track.
          if contains(file,'TRACK')
             strArray = string.empty;
             linestart = 0;
              % Update the row number of the storage matrix
              n_track = n_track + 1;
              % The extracted strings are stored in a tuple
              track_name = textscan(file,'%s');
              
              % Need to concatenate it into the flight number%%%%%
              track_name = strcat(track_name{1,1}{1,1},track_name{1,1}{2,1});
              
              
              % storage
              Total(n_track).flight = track_name;
              
              if   flag ~= n_track
                  
                  % Assign the point data to the previous row
                  Total(n_track-1).track = every_lt6;
                  % Reset Parameters
                  every_lt6 = {};
                  row = 1;
                  flag = flag + 1;
              end
              
              % Update the total line number of LT6
              lines = lines + 1;
          
              
           
          % Only the row of the 3D coordinate point has a row number, the rest do not, so the judgment condition is whether there is a comma
          elseif contains(file,',')
              
              infoChars = cellstr(strArray);
              Total(n_track).info = infoChars;

                  % Read the 3D coordinates of each row     x y z v t
                  content = textscan(file,'%d %d %d %d %d','Delimiter',',');
                  
                  % Take each dimension of coordinates from the cell and put them into the matrix
                  for i = 1:5
                      
                      every_lt6{row,i} = content{1,i};
                      
                  end
 
                  % Update the row number of the storage matrix
                  row = row + 1;
 
                  % Update the total line number of LT6
                  lines = lines + 1;
          else
              
                  % Update line number
                  lines = lines + 1;
                  linestart = linestart+1;
                  strArray(linestart) = file;

%                   line_star = line_star+1;
          end
           
          
      end
       
      Total(n_track).track = every_lt6;
       
      % Close File
      fclose(file_id);
      
end
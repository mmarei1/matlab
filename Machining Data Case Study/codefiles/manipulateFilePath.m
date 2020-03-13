function pathout = manipulateFilePath(pathin,newPath)
    pathout = pathin;
    substring_to_delete = "_newImage"
    for i=1:size(pathin)
        % select each image path
        substring_to_change = pathin(i);
        substring_man = replaceBetween(substring_to_change,...
            "/home","NewImages/EdgeNew/",newPath,'Boundaries','Inclusive')
        substring_man = strrep(substring_man,"/","\")
        substring_man = erase(substring_man,substring_to_delete)
        pathout(i) = substring_man
    end
end
function big_log = loc2bigloc(loc)
    big_log = [loc(:,1)-loc(:,3)/2, loc(:,2)-loc(:,4)/2, loc(:,3)*2, loc(:,4)*2];
end


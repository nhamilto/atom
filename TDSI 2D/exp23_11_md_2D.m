function f=exp23_11_md_2D(x_vec,path,lv,xi,yj);
x=path.x0+x_vec*path.cos;
y=path.y0+x_vec*path.sin;
f=exp23B_11(x,y,xi,yj,lv);
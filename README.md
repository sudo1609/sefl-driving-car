# sefl-driving-car
Download phần mềm mô phỏng tại: https://github.com/udacity/self-driving-car-sim

Link paper tham khảo mô hình self-driving car: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

File models.py là file chứa model CNN.

File augmentation.py là file dùng tạo data augmentation bằng OPENCV

File Run.py là file sử dụng để chạy mô hình.

Vào phần Training mode của phầm mềm tạo data sau khi đã tạo được data thì chạy tiếp file models.py. Sau khi train được models.py rồi sẽ
thu được một file model.h5.

Vào lại phầm mềm Chọn autonomous mod. Tiếp đến mở command line chạy python3 Run.py model.h5.


# author:aqizhou
# edit time:2022/10/29 11:15
import xlwt


#  将数据写入新文件
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, data[j])
        i = i + 1

    f.save(file_path)  # 保存文件


if __name__ == '__main__':
    file_path = "test_excel.xls"
    datas =[[1,2,3,4],[5,6,7,8]]
    data_write(file_path,datas)

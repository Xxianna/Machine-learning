//测试cmdsheet的控制命令
#include <iostream>
#define __windows__
#include "cmdsheet.h"


using namespace std;

int main(){
    //设置为unicode编码，否则中文会乱码
    CMD_INIT
    //测试各种颜色
    cout << "测试各种颜色：" << endl;
    cout<<"字色："<<CMD_BLUE<<"蓝色"<<CMD_RED<<"红色"<<CMD_GREEN<<"绿色"<<CMD_YELLOW<<"黄色"<<CMD_PURPLE<<"紫色"<<CMD_CYAN<<"青色"<<CMD_WHITE<<"白色"<<CMD_BLACK<<"黑"<<CMD_RESET<<"色"<<endl;
    cout<<"背景："<<CMD_BLUE_B<<"蓝色"<<CMD_RED_B<<"红色"<<CMD_GREEN_B<<"绿色"<<CMD_YELLOW_B<<"黄色"<<CMD_PURPLE_B<<"紫色"<<CMD_CYAN_B<<"青色"<<CMD_WHITE_B<<"白"<<CMD_BLACK_B<<"色黑色"<<CMD_RESET<<"色"<<endl;

    //测试光标移动
    cout << "测试光标移动：" << endl;
    cout << CMD_ROW_COL(10,10) << "光标移动到10行10列" << endl;
    cout << CMD_ROW(11) << "光标移动到11行" << endl;
    cout << CMD_COL(20) << "光标移动到20列" << endl;
    cout<<CMD_NEXT_LINE<<"光标移动到下一行"<<endl;
    cout<<CMD_PREV_LINE<<CMD_PREV_LINE<<"光标移动到上两行"<<endl;
    printf("\033[2A AAAA"); //光标上移两行
    //测试光标闪烁
    cout << "测试光标闪烁：" << endl;
    cout << CMD_CURSOR_SHOW << "显示光标" << endl;
    _sleep(1000);
    cout << CMD_CURSOR_HIDE << "隐藏光标" << endl;
    _sleep(1000);
    system("pause");

}
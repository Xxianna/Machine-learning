//文件通过合适的控制符，使得在控制台输出以进行各种格式的控制
//在windows中，需要设定命令行为unicode编码，否则无法控制
//linux下无需设定，直接使用即可
//请在包含前定义__windows__或__linux__，否则会报错
//为了可读性，建议移动光标统一使用CMD_ROW_COL(x,y)

//#define __windows__
// #define __linux__

#ifndef __windows__
#ifndef __linux__
#error "Please define __windows__ or __linux__"
#endif
#endif

#ifndef CMDSHEET_H
#define CMDSHEET_H

#ifdef __windows__
#define CMD_INIT        system("chcp 65001");system("cls"); //初始化，设定命令行为unicode编码
#else
#define CMD_INIT
#endif

#define CMD_RESET       "\033[0m"    //重置
#define CMD_BOLD        "\033[1m"    //加粗
#define CMD_UNDERLINE   "\033[4m"    //下划
#define CMD_BLINK       "\033[5m"    //闪烁
#define CMD_REVERSE     "\033[7m"    //反显
#define CMD_HIDE        "\033[8m"    //隐藏

#define CMD_BLACK       "\033[30m"   //黑色
#define CMD_RED         "\033[31m"   //红色
#define CMD_GREEN       "\033[32m"   //绿色
#define CMD_YELLOW      "\033[33m"   //黄色
#define CMD_BLUE        "\033[34m"   //蓝色
#define CMD_PURPLE      "\033[35m"   //�?�?
#define CMD_CYAN        "\033[36m"   //青色
#define CMD_WHITE       "\033[37m"   //白色

#define CMD_BLACK_B     "\033[40m"   //黑色背景
#define CMD_RED_B       "\033[41m"   //红色背景
#define CMD_GREEN_B     "\033[42m"   //绿色背景
#define CMD_YELLOW_B    "\033[43m"   //黄色背景
#define CMD_BLUE_B      "\033[44m"   //蓝色背景
#define CMD_PURPLE_B    "\033[45m"   //�?色背�?
#define CMD_CYAN_B      "\033[46m"   //青色背景
#define CMD_WHITE_B     "\033[47m"   //白色背景

//移动光标
#define CMD_UP(x)           "\033["#x"A" //上移x�?
#define CMD_DOWN(x)         "\033["#x"B" //下移x�?
#define CMD_RIGHT(x)        "\033["#x"C" //右移x�?
#define CMD_LEFT(x)         "\033["#x"D" //左移x�?
#define CMD_NEXT_LINE       "\033[1E"    //下一�?
#define CMD_PREV_LINE       "\033[1F"    //上一�?
#define CMD_ROW_COL(x,y)    "\033["#x";"#y"H" //移动到�?�x行�?�y�?
#define CMD_ROW(x)          "\033["#x";H" //移动到�?�x�?
#define CMD_COL(y)          "\033["#y"l" //移动到�?�y�?
#define CMD_CLEAR           "\033[2J"    //清屏
#define CMD_CLEAR_LINE      "\033[2K"    //清除从光标到行尾的内�?
#define CMD_CLEAR_LINE_B    "\033[1K"    //清除从光标到行�?�的内�??
#define CMD_CLEAR_LINE_A    "\033[0K"    //清除光标所在�?�的内�??
#define CMD_CLEAR_UP        "\033[1J"    //清除从光标到屏幕顶部的内�?
#define CMD_CLEAR_DOWN      "\033[0J"    //清除从光标到屏幕底部的内�?

//保存光标位置
#define CMD_SAVE_CURSOR     "\033[s"
//恢�?�光标位�?
#define CMD_RESTORE_CURSOR  "\033[u"

//设置光标样式
#define CMD_CURSOR_HIDE     "\033[?25l"
#define CMD_CURSOR_SHOW     "\033[?25h"

#endif
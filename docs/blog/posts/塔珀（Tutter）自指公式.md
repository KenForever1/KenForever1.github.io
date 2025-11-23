---
title: 塔珀（Tutter）自指公式是什么？
date: 2025-11-23
authors: [KenForever1]
categories: 
  - 杂谈
labels: []
comments: true
---

## Tutter公式并不高深莫测

今天，学习Tracy的时候介绍例程讲到了这个公式，当时很迷糊，觉得这个公式高深莫测。然后查阅资料学习了一下原理，分享记录一下。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/tutter-formula.png)

图片上方就是塔珀（Tutter）自指公式，意思就是给他一个特定的数字K，然后通过这个公式计算x值和y值坐标点的值，可以得到17*106像素格子的黑白图片。（0表示白色，1表示黑色），可以画出这个公式。也就是图片下方的渲染图，是不是很神奇！！

不看数学原理如何巧妙的构造这个公式，我们只看公式里面有17乘以x，y模运算17。这不就是坐标点的移动吗？

整个图片就是17列乘以106行的点格构成，我们只要在这个点格上画出来我们想要的图片，再把点格上的数字按照先行后列的顺序组成一个数字M，再乘以17就可以得到K值。

看个例子就明白了：
![](https://raw.githubusercontent.com/KenForever1/CDN/main/tutter_1.png)


![](https://raw.githubusercontent.com/KenForever1/CDN/main/tutter_2.png)

[wikipedia介绍里面的一张图](https://en.wikipedia.org/wiki/Tupper%27s_self-referential_formula)：

![](https://raw.githubusercontent.com/KenForever1/CDN/main/tutter_4.png)


首先有几点：

+ 公式左边的1/2实际上可以是0到1之间任意数，作者采用1/2这个值让公式也多蒙上了一个迷雾。因为对2取模运算，只会出现0和1两种结果，也就是黑白两种颜色。

+ 这个公式中的17就是图片的高度，106就是图片的宽。K值其实保存了图片的二进制数据信息，利用这个公式可以还原出每个坐标点的颜色值。

+ 如果你要画更大的图片，超过17x106，你也可以更改公式，K值范围也会变大。

+ 我们可以根据想要的图片，计算出不同的K值，就可以渲染不同的图片。比如：

不同的K值可以画出不同的效果(这里提供的K值按照公式画出来是倒着的镜像图片，所以代码里面y = height-y-1上下翻转了一下)。可以执行下python代码，也就可以把K值复制到网站里面[keelyhill.github.io/tuppers-formula](https://keelyhill.github.io/tuppers-formula/)。

![](https://raw.githubusercontent.com/KenForever1/CDN/main/tutter_3.png)

```python
# https://github.com/UlrikHjort/Tuppers-self-referential-formula/blob/master/Tupper.py
# Tupper
k1=4858450636189713423582095962494202044581400587983244549483093085061934704708809928450644769865524364849997247024915119110411605739177407856919754326571855442057210445735883681829823754139634338225199452191651284348332905131193199953502413758765239264874613394906870130562295813219481113685339535565290850023875092856892694555974281546386510730049106723058933586052544096664351265349363643957125565695936815184334857605266940161251266951421550539554519153785457525756590740540157929001765967965480064427829131488548259914721248506352686630476300


# Pacman
k2=144520248970897582847942537337194567481277782215150702479718813968549088735682987348888251320905766438178883231976923440016667764749242125128995265907053708020473915320841631792025549005418004768657201699730466383394901601374319715520996181145249781945019068359500510657804325640801197867556863142280259694206254096081665642417367403946384170774537427319606443899923010379398938675025786929455234476319291860957618345432248004921728033349419816206749854472038193939738513848960476759782673313437697051994580681869819330446336774047268864


# Euler
k3=2352035939949658122140829649197960929306974813625028263292934781954073595495544614140648457342461564887325223455620804204796011434955111022376601635853210476633318991990462192687999109308209472315419713652238185967518731354596984676698288025582563654632501009155760415054499960

# Assign k1,k2, k3 to k to get desired image
k = k1
width = 106
height = 17
scale = 5

fname = "foo"
image  = Image.new("RGB", (width, height),(255, 255, 255))

for x in range (width):
    for y in range (height):
        if ((k+y)//17//2**(17*int(x)+int(y)%17))%2 > 0.5:
            # Image need to be flipped vertically - therefore y = height-y-1
            image.putpixel((x, height-y-1), (0,0,0))


#scale up image
image = image.resize((width*scale,height*scale))
image.save(fname+".png")
```

## 参考资料

+ https://www.zhihu.com/question/35867114

+ https://en.wikipedia.org/wiki/Tupper%27s_self-referential_formula

+ https://keelyhill.github.io/tuppers-formula/?

+ https://github.com/UlrikHjort/Tuppers-self-referential-formula/blob/master/Tupper.py
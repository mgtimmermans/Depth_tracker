# Depth Tracking

 *"Diagram A represents a very 'grey' area for judges. The time that a lifter will remain at this position for a judge to gauge depth is fractional. Judging is ultimately a subjective venture that is governed by objective standards. Given he visual representation of these diagrams the above squat would be +/-50% chance of passage.*

 *If minimum depth (shown in Diagram A) is what a lifter chooses to squat to, they are risking the human error factor that is unavoidable given the current judging methodology used by the sport"* - [USAPL](https://archive.ph/h3lQe)

![alt text](Assets/usapl.png)

### Goal

To clear up the ambiguity on failed lifts using machine vision.


## Official Squat Rules (IPF Technical Rules 2023)
1. The lifter shall face the front of the platform. The bar shall be held horizontally across the shoulders, hands and fingers gripping the bar. The hands may be positioned anywhere on the bar inside and or in contact with the inner collars.

2. After removing the bar from the racks, (the lifter may be aided in removal of the bar from the racks by the spotter / loaders) the lifter must move backwards to establish the starting position. When the lifter is motionless, erect (slight deviation is allowable) with knees locked the Chief Referee will give the signal to begin the lift. The signal shall consist of a downward movement of the arm and the audible command “Squat”. Before receiving the signal to “squat” the lifter may make any position adjustments within the rules, without penalty. For reasons of safety the lifter will be requested to “Replace” the bar, together with a backward movement of the arm, if after a period of five seconds he is not in the correct position to begin the lift. The Chief Referee will then convey the reason why the signal was not given.

3. Upon receiving the Chief Referee’s signal the lifter must bend the knees and lower the body until the top surface of the legs at the hip joint is lower than the top of the knees. Only one decent attempt is allowed. The attempt is deemed to have commenced when the lifters knees have unlocked.

4. The lifter must recover at will to an upright position with the knees locked. Double bouncing at the bottom of the squat attempt or any downward movement is not permitted. When the lifter is motionless (in the apparent final position) the Chief Referee will give the signal to rack the bar.

5. The signal to rack the bar will consist of a backward motion of the arm and the audible command “Rack”. The lifter must then return the bar to the racks. Foot movement after the rack signal will not be cause for failure. For reasons of safety the lifter may request the aid of the spotter/loaders in returning the bar to, and replacing it in the racks. The lifter must stay with the bar during this process.
6. Not more than five and not less than two spotter/loaders shall be on the platform at any time. The Referees may decide to the number of spotter/loaders required on the platform at any time 2, 3, 4, or 5.

## Lights
A system of lights shall be provided whereby the referees make known their decisions. Each referee will control a white and a red light. These two colors represent a “good lift” and “no lift” respectively. The lights shall be arranged horizontally to correspond with the positions of the three referees.
`white_light = "good lift"` and `red_light = "no lift"`

## Causes for Disqualification

### Red card
* Failure to bend the knees and lower the body until the top surface of the legs (upper quad) at the hip joint is lower than the top of the knees. `model leg and measure`

### Blue card
* Failure to assume an upright position with the knees locked at the commencement and at the completion of the lift (you need to have your knees locked at both the beginning and end of the movement). `track fully extended length`
* Double bouncing at the bottom of the lift, or any downward movement during the ascent.
`track the bar coordinates`

### Yellow Card
* Stepping backward or forward or moving the feet laterally (you can't lose your balance). `track foot movement`
* Failure to observe the Chief Referees signals at the commencement or completion of the lift. `wait for lift commands`
* Contact with bar or lifter by the spotters/loaders between the Chief referees’ signals, in order to make the lift easier. `no external help`
* Contact of elbows or upper arms with the legs, which has supported and been of aid to the lifter. Slight contact that is of no aid may be ignored.
* Any dropping or dumping of the bar after completion of the lift. `rack the weight`
* Failure to comply with any of the requirements contained in the general description of the lift, which precedes this list of disqualification. `official squat rules`
* Incomplete lift. `failure`

## Process
1. Front judge says `squat` then once complete says `rack`.


## Judge notes
* 1 front judge, 2 side judges (3 lights)
* Side judges are the ones primarily responsible for judging squat depth
* Camera angle matters, side view, perpendicular, knee height
* Dont let quads fool you, the larger the quads, the shallower the squat will look


A Judge can be seen in a chair to the right (red shirt)
![Judge](Assets/judge_1.png)
Source: [EliteFTS](https://www.elitefts.com/education/from-the-judges-chair-the-squat/)

USPA judge Jeffrey Winkler finding a position for the best view of the squat from his right-side place. USPA Hardknox Third Annual Powerlifting Meet, 6/17/17, Brownsville TX
![USAPLJUDGE](Assets/judge_2.png)
Source: [EliteFTS](https://www.elitefts.com/education/from-the-judges-chair-the-squat/)


## Squat notes
Deep squat: 120° knee flexion or more
![top of knee](Assets/angle.png)

The following picture illustrates what the ‘top of the leg' below the ‘top of the knee' means:
![top of knee](Assets/top_knee.png)

highest part of the knee vs highest part of the hip
![top of knee](Assets/top_knee_2.png)

First image is failure, bottom image is good lift
![top of knee](Assets/leg.png)



#### Failure Cards
After the lights have been activated and appeared, the referee(s) will raise a card or paddle or activate a light system to make known the reason/s why the lift has been failed.


```C++
if (red_light) {
    Case (red_card){...}
    Case (blue_card){...}
    Case (yellow_card){...}
}
```






# Bench
stretch goal or for someone to contribute towards!

### Bench Requirements
The measurements and requirements of the actual bench
* `length > 122` : L not less than 1.22 m
* `29 > width > 32` : W not less than 29 cm and not exceeding 32 cm 
* `42 > height > 45` : H not less than 42 cm and not exceeding 45 cm measured from the floor to the top of the padded surface

### Good lift examples

![good bench 1](Assets/bench_good_1.png)
![good bench 2](Assets/bench_good_2.png) 
Source: [IPF Technical Rules](IPF_Technical_Rules_Book_2023__1_.pdf)
### Bad lift examples

![Bad bench 1](Assets/bench_bad_1.png)
![Bad bench 2](Assets/bench_bad_2.png)
Source: [IPF Technical Rules](IPF_Technical_Rules_Book_2023__1_.pdf)






## Source

#### Powerlifting
* [IPF Technical Rules 2023](https://www.powerlifting.sport/fileadmin/ipf/data/rules/technical-rules/english/IPF_Technical_Rules_Book_2023__1_.pdf)
* [temple of iron](https://www.temple-of-iron.com/squat-depth-are-you-deep-enough/)
* [strengthlog](https://www.strengthlog.com/squat-depth-how-deep-should-you-squat/)
* [Powerlifting technique](https://powerliftingtechnique.com/powerlifting-squat-technique-rules/)

#### Software
* https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

* https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/






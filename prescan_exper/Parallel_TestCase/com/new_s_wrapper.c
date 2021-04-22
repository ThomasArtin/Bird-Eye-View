
/*
 * Include Files
 *
 */
#if defined(MATLAB_MEX_FILE)
#include "tmwtypes.h"
#include "simstruc_types.h"
#else
#include "rtwtypes.h"
#endif



/* %%%-SFUNWIZ_wrapper_includes_Changes_BEGIN --- EDIT HERE TO _END */
#include <math.h>
/* %%%-SFUNWIZ_wrapper_includes_Changes_END --- EDIT HERE TO _BEGIN */
#define u_width 1
#define y_width 1

/*
 * Create external references here.  
 *
 */
/* %%%-SFUNWIZ_wrapper_externs_Changes_BEGIN --- EDIT HERE TO _END */
/* extern double func(double a); */
/* %%%-SFUNWIZ_wrapper_externs_Changes_END --- EDIT HERE TO _BEGIN */

/*
 * Output functions
 *
 */
void new_s_Outputs_wrapper(const real_T *Front,
			const real_T *LEFT,
			const real_T *RIGHT,
			const real_T *BACK,
			const real_T *current_v,
			const real_T *X,
			const real_T *Y,
			const real_T *parallel_r,
			real_T *steering,
			real_T *velocity,
			real_T *brake,
			real_T *Trottle,
			real_T *counter,
			real_T *counter1,
			real_T *counter2,
			real_T *counter3,
			real_T *counter4,
			real_T *autog)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */
if(*counter4 == 0){
            if (*RIGHT != 0 && *counter3 == 0){
            *velocity = 10;
            *brake = 0;
            *steering = 0;
            *Trottle = 1;
        }
        else{
            *counter3 = 1 ;
            *steering = 0;
            *brake = 90;
            *velocity = 0;
            if(*current_v > 0 && *current_v<0.2){
                        *counter4 = 1;
            }

        }
}
else{
    if(*BACK <1.8 && *BACK !=0 && *counter2 == 0  ){
            *brake = 150;
            *velocity = 0;
            *Trottle = 0;
            *counter1 = 0;
            *counter = 1;
            if(*current_v > 0 && *current_v < 1){
                *counter1 = 1;
                *brake = 0;
                *Trottle = 2;
                *autog = 1;
                *steering = -700;
                *velocity = 5;
            }
        }
        else if(*counter1 == 1 && *BACK >0.3 ) {
            *counter2 = 1;
            *brake = 200;
            *velocity = 0;
            if(*current_v > 0 && *current_v < 1){
                 *brake = 0;
                *counter = 1;
                *Trottle = 5;
                *velocity = -5;
                *autog = -1;
                *steering = 770;
                if(*BACK <1.5){
                    *steering -= 20;
                }
           }

         }
        else if(*counter == 0){
                *counter = 0;
                counter1 = 0;
                *brake = 0;
                *steering = -360;
		*autog=-1;
                if(*BACK <3){
                    *steering = *steering +10;
                }
                 *velocity = -5;
                *Trottle = 1; 
        }
        else {
                 *steering = 0;
                  *brake = 150;
                   *velocity = 0;
        }
  

}
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}



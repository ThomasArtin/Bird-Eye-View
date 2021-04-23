
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
void ssss_Outputs_wrapper(const real_T *Front,
			const real_T *LEFT,
			const real_T *RIGHT,
			const real_T *BACK,
			const real_T *current_v,
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
			real_T *counter5,
			real_T *autog)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */
double epison = 0.0001;
if(*parallel_r == 1){
    //first BACK
    if(*counter1 == 0 ){
        if((*BACK == 0 || *BACK >1.3)  && *counter == 0){
            *steering = -370;
            *velocity = 5;
            *brake = 0;
            *autog = -1;
            *Trottle = 1;
        
        }
         else{
            *counter = 1;
            *steering = 0;
            *velocity = 0;
            *brake = 150;
            *autog = -1;
            *Trottle = 0;
            if(*current_v < epison ){
                *counter2 = 1;
            }
 
        }
    }
    //First RIGHT 
    if(*counter2 == 1 ){
        *counter1 = 1;
        if(*BACK > 3){
              *steering = 0;
               *brake = 150;
              *velocity = 0;
              *autog = -1;
              *Trottle = 0;
              if(*current_v < epison ){
                    *counter3 = 1;
                }
        }
        else{
            *steering = -180;
            *velocity = 5;
            *brake = 0;
            *autog = 1;
            *Trottle = 1;
            
        }         
    }  
    //second BACK
    if(*counter3 == 1){
        *counter2 = 0;
        if(*BACK < 0.5){
            *steering = 0;
            *velocity = 0;
            *brake = 150;
            *autog = 1;
            *Trottle = 1;
            if(*current_v < epison ){
                *counter4 = 1;
            }
        }
        else{
            *steering = 300;
            *velocity = -5;
            *brake = 0;
            *autog = -1;
            *Trottle = 1;
            
        }
        
    }
    //Forward Right
    if(*counter4 ==1){
        counter3 = 0;
          if(*BACK > 1.6 ){
            *steering = 0;
            *velocity = 0;
            *brake = 150;
            *autog = 1;
            *Trottle = 0;
            if(*current_v < epison ){
                *counter5 = 1;
            }
        }
        else{
            *steering = -700;
            *velocity = 5;
            *brake = 20;
            *autog = 1;
            *Trottle = 1;
        }
        
        
    }
    if(*counter5 == 1){
            *counter4 =0;
              if(*BACK < 0.5 ){
                *steering = 0;
                *velocity = 0;
                *brake = 150;
                *autog = 1;
                *Trottle = 0;
                if(*current_v < epison ){
                    *counter5 = 1;
                }
            }
            else{
                *steering = 0;
                *velocity = -5;
                *brake = 0;
                *autog = -1;
                *Trottle = 1;
            }
        
    }
}
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}



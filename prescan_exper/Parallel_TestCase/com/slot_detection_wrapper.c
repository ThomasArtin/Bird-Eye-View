
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
void slot_detection_Outputs_wrapper(const real_T *Front,
			const real_T *LEFT,
			const real_T *RIGHT,
			const real_T *current_v,
			real_T *steering,
			real_T *velocity,
			real_T *brake,
			real_T *Trottle,
			real_T *autog,
			real_T *parallel_r,
			real_T *parallel_l,
			real_T *perp_r,
			real_T *perp_l,
			real_T *counter3,
			real_T *direction)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */
/*if(Front != 0){
    *brake = 150;
    *velocity = 0;
    *Trottle = 0;
}*/
if(*parallel_r == 0){
    if (*RIGHT != 0 && *counter3 == 0){
            *velocity = 10;
            *brake = 0;
            *steering = 0;
            *Trottle = 1;
            *autog = 1;
        }
else{
        *counter3 = 1 ;
        *steering = 0;
        *brake = 100;
        *velocity = 0;
         *parallel_l = -1;
        if(*current_v > 0 && *current_v <0.2){
                    *parallel_r = 1;
                   
        }   
    }
}
if(*parallel_l == 0){
    //*counter3 = 0;
    if (*LEFT != 0 && *counter3 == 0){
            *velocity = 10;
            *brake = 0;
            *steering = 0;
            *Trottle = 1;
            *autog = 1;
        }
    else{
        *counter3 = 1 ;
        *steering = 0;
        *brake = 100;
        *velocity = 0;
        *parallel_r = -1;
        if(*current_v > 0 && *current_v <0.2){
                    *parallel_l = 1;
                    
         }   
        }
}
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}



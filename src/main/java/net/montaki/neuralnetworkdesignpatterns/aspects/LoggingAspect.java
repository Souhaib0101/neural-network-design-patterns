package net.montaki.neuralnetworkdesignpatterns.aspects;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.DeclarePrecedence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Aspect
@DeclarePrecedence("net.montaki.neuralnetworkdesignpatterns.aspects.AuthorizationAspect, "
        + "net.montaki.neuralnetworkdesignpatterns.aspects.LoggingAspect, *")
public class LoggingAspect {
    private static final Logger LOGGER = LoggerFactory.getLogger(LoggingAspect.class);

    // Les identifiants et les arguments ne sont jamais journalises.
    @Around("execution(* net.montaki.neuralnetworkdesignpatterns..*(..))"
            + " && !within(net.montaki.neuralnetworkdesignpatterns.aspects..*)"
            + " && !within(net.montaki.neuralnetworkdesignpatterns.security..*)")
    public Object log(ProceedingJoinPoint joinPoint) throws Throwable {
        if (!LOGGER.isDebugEnabled()) {
            return joinPoint.proceed();
        }
        String method = joinPoint.getSignature().toShortString();
        long start = System.nanoTime();
        LOGGER.debug("AVANT {}", method);
        try {
            return joinPoint.proceed();
        } catch (Throwable failure) {
            LOGGER.debug("ECHEC {} : {}", method, failure.getClass().getSimpleName());
            throw failure;
        } finally {
            LOGGER.debug("APRES {} duree={} ms", method, (System.nanoTime() - start) / 1_000_000.0);
        }
    }
}
